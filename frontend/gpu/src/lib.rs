use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU32, Ordering},
};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

// Two motion vectors, four palette colours and a depth-four expression tree.
const UNIFORM_FLOATS: usize = (2 + 4 + 31) * 4;
const MAX_SIDE: u32 = 192;
// One frame can render while the previous completion crosses the browser's IPC.
const MAX_PENDING_FRAMES: u32 = 2;

#[wasm_bindgen]
pub struct Gpu {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    failed: Arc<AtomicBool>,
}

#[wasm_bindgen]
impl Gpu {
    pub async fn create() -> Result<Gpu, JsValue> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                ..Default::default()
            })
            .await
            .map_err(js_error)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Slate live throbber"),
                ..Default::default()
            })
            .await
            .map_err(js_error)?;
        let failed = Arc::new(AtomicBool::new(false));
        let lost = failed.clone();
        device.set_device_lost_callback(move |_, _| {
            lost.store(true, Ordering::Relaxed);
        });
        let invalid = failed.clone();
        device.on_uncaptured_error(Arc::new(move |_| {
            invalid.store(true, Ordering::Relaxed);
        }));
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            failed,
        })
    }

    pub async fn renderer(&self, canvas: HtmlCanvasElement) -> Result<Renderer, JsValue> {
        if self.failed.load(Ordering::Relaxed) {
            return Err(js_error("GPU unavailable"));
        }
        let surface = self
            .instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(js_error)?;
        let mut config = surface
            .get_default_config(&self.adapter, 64, 64)
            .ok_or_else(|| js_error("Canvas surface unavailable"))?;
        // wgpu 28's browser capabilities list only Opaque; its WebGPU configure
        // path also supports PreMultiplied, which keeps the circle edge clear.
        config.alpha_mode = wgpu::CompositeAlphaMode::PreMultiplied;
        surface.configure(&self.device, &config);
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Bounded Random Art"),
                source: wgpu::ShaderSource::Wgsl(include_str!("art.wgsl").into()),
            });
        let pipeline = pipeline(&self.device, &shader, config.format);
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Art parameters"),
            size: (UNIFORM_FLOATS * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let binding = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        if let Some(error) = validation.pop().await {
            return Err(js_error(error));
        }
        Ok(Renderer {
            surface,
            config,
            device: self.device.clone(),
            queue: self.queue.clone(),
            pipeline,
            uniform,
            binding,
            failed: self.failed.clone(),
            pending: Arc::new(AtomicU32::new(0)),
        })
    }
}

#[wasm_bindgen]
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    uniform: wgpu::Buffer,
    binding: wgpu::BindGroup,
    failed: Arc<AtomicBool>,
    pending: Arc<AtomicU32>,
}

#[wasm_bindgen]
impl Renderer {
    pub fn resize(&mut self, side: u32) {
        self.config.width = side.clamp(1, MAX_SIDE);
        self.config.height = self.config.width;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn render(&self, parameters: &[f32]) -> Result<bool, JsValue> {
        if parameters.len() != UNIFORM_FLOATS || parameters.iter().any(|x| !x.is_finite()) {
            return Err(js_error("Invalid art parameters"));
        }
        if self.failed.load(Ordering::Relaxed) {
            return Err(js_error("GPU device lost"));
        }
        // Slow adapters must not accumulate an unbounded queue of stale images.
        if self.pending.load(Ordering::Relaxed) >= MAX_PENDING_FRAMES {
            return Ok(false);
        }
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Timeout) => return Ok(false),
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.config);
                return Ok(false);
            }
            Err(error) => return Err(js_error(error)),
        };
        self.queue
            .write_buffer(&self.uniform, 0, bytemuck::cast_slice(parameters));
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.binding, &[]);
            pass.draw(0..3, 0..1);
        }
        self.pending.fetch_add(1, Ordering::Relaxed);
        self.queue.submit([encoder.finish()]);
        let pending = self.pending.clone();
        self.queue.on_submitted_work_done(move || {
            pending.fetch_sub(1, Ordering::Relaxed);
        });
        frame.present();
        Ok(true)
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.uniform.destroy();
    }
}

fn pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Random Art circle"),
        layout: None,
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vertex"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fragment"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

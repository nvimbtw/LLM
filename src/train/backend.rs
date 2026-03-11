use wgpu::util::DeviceExt;
use std::sync::{Arc, Mutex};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MatrixDimensions {
    a_rows: u32, a_cols: u32, b_cols: u32, batch_size: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
struct Dimensions {
    rows: u32, cols: u32, scale: f32, padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
struct OpsDimensions {
    len: u32, padding1: u32, padding2: u32, padding3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
struct UpdateParams {
    len: u32, lr: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
struct AdamParams {
    len: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    correction1: f32,
    correction2: f32,
    padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
struct EmbeddingParams {
    vocab_size: u32,
    dimensions: u32,
    seq_len: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Default)]
struct LossParams {
    batch_size: u32,
    vocab_size: u32,
    seq_len: u32,
    padding: u32,
}

pub struct GpuBufferPool {
    buffers: Mutex<HashMap<(u64, u32), Vec<wgpu::Buffer>>>,
}

impl GpuBufferPool {
    pub fn new() -> Self {
        Self { buffers: Mutex::new(HashMap::new()) }
    }

    pub fn get(&self, device: &wgpu::Device, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let mut buffers = self.buffers.lock().unwrap();
        let key = (size, usage.bits());
        if let Some(list) = buffers.get_mut(&key) {
            if let Some(buf) = list.pop() {
                return buf;
            }
        }
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size, usage, mapped_at_creation: false,
        })
    }

    pub fn return_buffer(&self, buf: wgpu::Buffer, usage: wgpu::BufferUsages) {
        let size = buf.size();
        let mut buffers = self.buffers.lock().unwrap();
        let key = (size, usage.bits());
        buffers.entry(key).or_insert_with(Vec::new).push(buf);
    }
}

pub struct WgpuBackend {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub pool: Arc<GpuBufferPool>,
    pub matmul_pipeline: wgpu::ComputePipeline,
    pub relu_pipeline: wgpu::ComputePipeline,
    pub softmax_pipeline: wgpu::ComputePipeline,
    pub add_pipeline: wgpu::ComputePipeline,
    pub add_assign_pipeline: wgpu::ComputePipeline,
    pub transpose_pipeline: wgpu::ComputePipeline,
    pub update_pipeline: wgpu::ComputePipeline, // For f32 grads
    pub update_fixed_pipeline: wgpu::ComputePipeline, // For i32 fixed-point grads
    pub adam_pipeline: wgpu::ComputePipeline, // For Adam optimizer
    pub scale_pipeline: wgpu::ComputePipeline, // Re-added scale_pipeline
    pub softmax_backward_pipeline: wgpu::ComputePipeline,
    pub scale_mask_pipeline: wgpu::ComputePipeline,
    pub relu_backward_pipeline: wgpu::ComputePipeline,
    pub layer_norm_pipeline: wgpu::ComputePipeline,
    pub layer_norm_backward_pipeline: wgpu::ComputePipeline,
    pub embedding_forward_pipeline: wgpu::ComputePipeline,
    pub embedding_backward_pipeline: wgpu::ComputePipeline,
    pub cross_entropy_pipeline: wgpu::ComputePipeline,
}

impl WgpuBackend {
    pub async fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None, force_fallback_adapter: false,
        }).await?;

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("LLM GPU Device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
        }, None).await.ok()?;

        let matmul_shader = device.create_shader_module(wgpu::include_wgsl!("../../shaders/matmul.wgsl"));
        let activations_shader = device.create_shader_module(wgpu::include_wgsl!("../../shaders/activations.wgsl"));
        let ops_shader = device.create_shader_module(wgpu::include_wgsl!("../../shaders/ops.wgsl"));
        let transpose_shader = device.create_shader_module(wgpu::include_wgsl!("../../shaders/transpose.wgsl"));
        let update_shader = device.create_shader_module(wgpu::include_wgsl!("../../shaders/update.wgsl"));
        let backprop_shader = device.create_shader_module(wgpu::include_wgsl!("../../shaders/backprop.wgsl"));
        let embedding_shader = device.create_shader_module(wgpu::include_wgsl!("../../shaders/embedding.wgsl"));
        let loss_shader = device.create_shader_module(wgpu::include_wgsl!("../../shaders/loss.wgsl"));

        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &matmul_shader, entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let relu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &activations_shader, entry_point: Some("relu_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let softmax_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &activations_shader, entry_point: Some("softmax_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &ops_shader, entry_point: Some("add_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let add_assign_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &ops_shader, entry_point: Some("add_assign_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let transpose_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &transpose_shader, entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &update_shader, entry_point: Some("main"), // f32 grads
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let update_fixed_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &update_shader, entry_point: Some("update_fixed_main"), // i32 fixed-point grads
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });
        
        let adam_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &update_shader, entry_point: Some("adam_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let scale_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &ops_shader, entry_point: Some("scale_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let softmax_backward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &backprop_shader, entry_point: Some("softmax_backward_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let scale_mask_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &backprop_shader, entry_point: Some("scale_mask_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let relu_backward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &backprop_shader, entry_point: Some("relu_backward_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let layer_norm_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &activations_shader, entry_point: Some("layer_norm_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let layer_norm_backward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &backprop_shader, entry_point: Some("layer_norm_backward_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let embedding_forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &embedding_shader, entry_point: Some("forward_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let embedding_backward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &embedding_shader, entry_point: Some("backward_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        let cross_entropy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, layout: None, module: &loss_shader, entry_point: Some("cross_entropy_forward_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(), cache: None,
        });

        Some(Self {
            device: Arc::new(device), queue: Arc::new(queue), pool: Arc::new(GpuBufferPool::new()),
            matmul_pipeline, relu_pipeline,
            softmax_pipeline, add_pipeline, add_assign_pipeline, transpose_pipeline, update_pipeline,
            update_fixed_pipeline, adam_pipeline,
            scale_pipeline,
            softmax_backward_pipeline, scale_mask_pipeline, relu_backward_pipeline,
            layer_norm_pipeline, layer_norm_backward_pipeline,
            embedding_forward_pipeline, embedding_backward_pipeline,
            cross_entropy_pipeline,
        })
    }

    pub fn run_matmul(&self, a: &GpuTensor, b: &GpuTensor) -> GpuTensor {
        self.run_batched_matmul(a, b, 1, a.shape.0, a.shape.1, b.shape.1)
    }

    pub fn run_batched_matmul(&self, a: &GpuTensor, b: &GpuTensor, batch_size: usize, m: usize, k: usize, n: usize) -> GpuTensor {
        let out_shape = if batch_size > 1 {
            // If batched, the output GpuTensor shape logic is tricky because GpuTensor is 2D.
            // We'll return (batch * m, n) which flattens the batch dimension.
            (batch_size * m, n)
        } else {
            (m, n)
        };
        
        let size = (batch_size * m * n * 4) as u64;
        let output_buffer = self.pool.get(&self.device, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
        
        let dims = MatrixDimensions { 
            a_rows: m as u32, 
            a_cols: k as u32, 
            b_cols: n as u32, 
            batch_size: batch_size as u32 
        };
        let dim_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.matmul_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dim_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output_buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.matmul_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            cp.dispatch_workgroups((m as u32 + 15) / 16, (n as u32 + 15) / 16, batch_size as u32);
        }
        self.queue.submit(Some(encoder.finish()));
        GpuTensor { buffer: output_buffer, shape: out_shape }
    }
    
    pub fn run_adam_update(&self, weights: &mut GpuTensor, grads: &GpuTensor, m: &mut GpuTensor, v: &mut GpuTensor, lr: f32, beta1: f32, beta2: f32, eps: f32, t: u32) {
        let total = (weights.shape.0 * weights.shape.1) as u32;
        let correction1 = 1.0 - beta1.powi(t as i32);
        let correction2 = 1.0 - beta2.powi(t as i32);
        
        let params = AdamParams { 
            len: total, 
            lr, 
            beta1, 
            beta2, 
            epsilon: eps, 
            correction1,
            correction2,
            padding: 0 
        };
        let param_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[params]), usage: wgpu::BufferUsages::UNIFORM,
        });
        
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.adam_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: param_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weights.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grads.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: m.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: v.buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.adam_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            Self::dispatch_flat(&mut cp, total, 64);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn run_relu(&self, data: &mut GpuTensor) {
        let dims = Dimensions { rows: data.shape.0 as u32, cols: data.shape.1 as u32, ..Default::default() };
        let dim_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.relu_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dim_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data.buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.relu_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            Self::dispatch_flat(&mut cp, (data.shape.0 * data.shape.1) as u32, 64);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn run_softmax(&self, data: &mut GpuTensor) {
        let dims = Dimensions { rows: data.shape.0 as u32, cols: data.shape.1 as u32, ..Default::default() };
        let dim_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.softmax_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dim_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data.buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.softmax_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            cp.dispatch_workgroups(data.shape.0 as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn run_add(&self, a: &GpuTensor, b: &GpuTensor) -> GpuTensor {
        let total = (a.shape.0 * a.shape.1) as u32;
        let size = (total * 4) as u64;
        let out_buffer = self.pool.get(&self.device, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);

        let dims = OpsDimensions { len: total, ..Default::default() };
        let dim_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.add_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dim_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.add_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            Self::dispatch_flat(&mut cp, total, 64);
        }
        self.queue.submit(Some(encoder.finish()));
        GpuTensor { buffer: out_buffer, shape: a.shape }
    }

    pub fn run_add_to_grad(&self, grad: &mut GpuTensor, d_w: &GpuTensor) {
        let total = (grad.shape.0 * grad.shape.1) as u32;
        let dims = OpsDimensions { len: total, ..Default::default() };
        let dim_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.add_assign_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dim_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grad.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: d_w.buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.add_assign_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            Self::dispatch_flat(&mut cp, total, 64);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn run_transpose(&self, input: &GpuTensor) -> GpuTensor {
        let out_shape = (input.shape.1, input.shape.0);
        let size = (input.shape.0 * input.shape.1 * 4) as u64;
        let output_buffer = self.pool.get(&self.device, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);

        let dims = [input.shape.0 as u32, input.shape.1 as u32];
        let dim_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&dims), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.transpose_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dim_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.transpose_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            cp.dispatch_workgroups((input.shape.0 as u32 + 15) / 16, (input.shape.1 as u32 + 15) / 16, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        GpuTensor { buffer: output_buffer, shape: out_shape }
    }

    pub fn run_update_f32(&self, weights: &mut GpuTensor, grads: &GpuTensor, lr: f32) {
        let total = (weights.shape.0 * weights.shape.1) as u32;
        let param_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[UpdateParams { len: total, lr }]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.update_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: param_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weights.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grads.buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.update_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            Self::dispatch_flat(&mut cp, total, 64);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn run_update_i32(&self, weights: &mut GpuTensor, grads_i32_buffer: &wgpu::Buffer, lr: f32) {
        let total = (weights.shape.0 * weights.shape.1) as u32;
        let param_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[UpdateParams { len: total, lr }]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.update_fixed_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: param_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weights.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: grads_i32_buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.update_fixed_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            Self::dispatch_flat(&mut cp, total, 64);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn run_scale(&self, data: &mut GpuTensor, scale: f32) {
        let total = (data.shape.0 * data.shape.1) as u32;
        let param_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[UpdateParams { len: total, lr: scale }]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.scale_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: param_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: data.buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.scale_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            Self::dispatch_flat(&mut cp, total, 64);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn run_softmax_backward(&self, probs: &GpuTensor, d_probs: &GpuTensor, scale: f32) -> GpuTensor {
        let size = (probs.shape.0 * probs.shape.1 * 4) as u64;
        let out_buffer = self.pool.get(&self.device, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);

        let dims = Dimensions { rows: probs.shape.0 as u32, cols: probs.shape.1 as u32, scale, padding: 0 };
        let dim_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.softmax_backward_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dim_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: probs.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: d_probs.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.softmax_backward_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            cp.dispatch_workgroups(probs.shape.0 as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        GpuTensor { buffer: out_buffer, shape: probs.shape }
    }

    pub fn run_scale_mask(&self, data: &mut GpuTensor, scale: f32) {
        let dims = Dimensions { rows: data.shape.0 as u32, cols: data.shape.1 as u32, scale, padding: 0 };
        let dim_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.scale_mask_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dim_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: data.buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.scale_mask_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            cp.dispatch_workgroups((data.shape.0 as u32 + 15) / 16, (data.shape.1 as u32 + 15) / 16, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn run_relu_backward(&self, input: &GpuTensor, d_output: &mut GpuTensor) {
        let dims = Dimensions { rows: input.shape.0 as u32, cols: input.shape.1 as u32, ..Default::default() };
        let dim_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.relu_backward_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dim_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: d_output.buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.relu_backward_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            Self::dispatch_flat(&mut cp, (input.shape.0 * input.shape.1) as u32, 64);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn run_layer_norm(&self, data: &GpuTensor) -> GpuTensor {
        let total_size = (data.shape.0 * data.shape.1 * 4) as u64;
        let out_buffer = self.pool.get(&self.device, total_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
        
        // We reuse the input buffer's data in the out_buffer initially
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&data.buffer, 0, &out_buffer, 0, total_size);
        self.queue.submit(Some(encoder.finish()));

        let dims = Dimensions { rows: data.shape.0 as u32, cols: data.shape.1 as u32, ..Default::default() };
        let dim_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.layer_norm_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dim_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.layer_norm_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            cp.dispatch_workgroups(data.shape.0 as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        GpuTensor { buffer: out_buffer, shape: data.shape }
    }

    pub fn run_layer_norm_backward(&self, normalized_input: &GpuTensor, d_output: &GpuTensor) -> GpuTensor {
        let total_size = (d_output.shape.0 * d_output.shape.1 * 4) as u64;
        let out_buffer = self.pool.get(&self.device, total_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);

        let dims = Dimensions { rows: d_output.shape.0 as u32, cols: d_output.shape.1 as u32, ..Default::default() };
        let dim_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.layer_norm_backward_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dim_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: normalized_input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: d_output.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.layer_norm_backward_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            cp.dispatch_workgroups(d_output.shape.0 as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        GpuTensor { buffer: out_buffer, shape: d_output.shape }
    }

    pub fn run_embedding_forward(&self, tokens: &wgpu::Buffer, embedding: &GpuTensor, positional: &GpuTensor, seq_len: usize) -> GpuTensor {
        let dims = embedding.shape.1;
        let out_shape = (seq_len, dims);
        let size = (seq_len * dims * 4) as u64;
        let out_buffer = self.pool.get(&self.device, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);

        let params = EmbeddingParams { vocab_size: embedding.shape.0 as u32, dimensions: dims as u32, seq_len: seq_len as u32, _padding: 0 };
        let param_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[params]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.embedding_forward_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: param_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tokens.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: embedding.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: positional.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: out_buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.embedding_forward_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            Self::dispatch_flat(&mut cp, (seq_len * dims) as u32, 64);
        }
        self.queue.submit(Some(encoder.finish()));
        GpuTensor { buffer: out_buffer, shape: out_shape }
    }

    pub fn run_embedding_backward(&self, tokens: &wgpu::Buffer, d_input: &GpuTensor, grad_emb_i32_buffer: &wgpu::Buffer, grad_pos: &mut GpuTensor) {
        let params = EmbeddingParams { vocab_size: (grad_emb_i32_buffer.size() / 4) as u32 / grad_pos.shape.1 as u32, dimensions: grad_pos.shape.1 as u32, seq_len: d_input.shape.0 as u32, _padding: 0 };
        let param_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[params]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.embedding_backward_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: param_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tokens.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: d_input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: grad_emb_i32_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: grad_pos.buffer.as_entire_binding() },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.embedding_backward_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            Self::dispatch_flat(&mut cp, (d_input.shape.0 * d_input.shape.1) as u32, 64);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn run_cross_entropy(&self, logits: &GpuTensor, target_token_ids: &[u32]) -> (GpuTensor, GpuTensor) {
        let batch_size = logits.shape.0;
        let vocab_size = logits.shape.1;

        let loss_output_buffer = self.pool.get(&self.device, (batch_size * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
        let grad_logits_batch_buffer = self.pool.get(&self.device, (logits.shape.0 * logits.shape.1 * 4) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);

        let params = LossParams {
            batch_size: batch_size as u32,
            vocab_size: vocab_size as u32,
            seq_len: batch_size as u32,
            padding: 0,
        };
        let param_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[params]), usage: wgpu::BufferUsages::UNIFORM,
        });

        let target_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(target_token_ids), usage: wgpu::BufferUsages::STORAGE,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.cross_entropy_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: param_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: logits.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: target_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: loss_output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grad_logits_batch_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            cp.set_pipeline(&self.cross_entropy_pipeline); cp.set_bind_group(0, &bind_group, &[]);
            cp.dispatch_workgroups(batch_size as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        (GpuTensor { buffer: loss_output_buffer, shape: (batch_size, 1) }, GpuTensor { buffer: grad_logits_batch_buffer, shape: logits.shape })
    }

    fn dispatch_flat(cp: &mut wgpu::ComputePass, total_elements: u32, workgroup_size: u32) {
        let total_workgroups = (total_elements + workgroup_size - 1) / workgroup_size;
        let x = total_workgroups.min(1024);
        let y = (total_workgroups + 1023) / 1024;
        cp.dispatch_workgroups(x, y, 1);
    }
}

pub struct GpuTensor {
    pub buffer: wgpu::Buffer,
    pub shape: (usize, usize),
}

impl GpuTensor {
    pub fn return_to_pool(self, backend: &WgpuBackend) {
        backend.pool.return_buffer(self.buffer, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
    }

    pub fn from_cpu(backend: &WgpuBackend, data: &Vec<Vec<f32>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        let size = (rows * cols * 4) as u64;

        let buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        for (i, row) in data.iter().enumerate() {
            let offset = (i * cols * 4) as u64;
            backend.queue.write_buffer(&buffer, offset, bytemuck::cast_slice(row));
        }

        Self { buffer, shape: (rows, cols) }
    }

    pub fn zero(&self, backend: &WgpuBackend) {
        let mut encoder = backend.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.clear_buffer(&self.buffer, 0, None);
        backend.queue.submit(Some(encoder.finish()));
    }
    
    // Custom zero for i32 buffer, for convenience
    pub fn zero_i32(buffer: &wgpu::Buffer, backend: &WgpuBackend) {
        let mut encoder = backend.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.clear_buffer(buffer, 0, None);
        backend.queue.submit(Some(encoder.finish()));
    }

    pub fn clone_on_gpu(&self, backend: &WgpuBackend) -> Self {
        let size = (self.shape.0 * self.shape.1 * 4) as u64;
        let new_buffer = backend.pool.get(&backend.device, size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
        let mut encoder = backend.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &new_buffer, 0, size);
        backend.queue.submit(Some(encoder.finish()));
        Self { buffer: new_buffer, shape: self.shape }
    }

    pub async fn to_cpu(&self, backend: &WgpuBackend) -> Vec<Vec<f32>> {
        let size = (self.shape.0 * self.shape.1 * 4) as u64;
        let staging = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let mut encoder = backend.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, size);
        backend.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        backend.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let flat: &[f32] = bytemuck::cast_slice(&data);
        let mut res = vec![vec![0.0f32; self.shape.1]; self.shape.0];
        for i in 0..self.shape.0 { for j in 0..self.shape.1 { res[i][j] = flat[i * self.shape.1 + j]; } }
        drop(data); staging.unmap();
        res
    }

    pub async fn last_row_to_cpu(&self, backend: &WgpuBackend) -> Vec<f32> {
        let row_size = (self.shape.1 * 4) as u64;
        let offset = ((self.shape.0 - 1) * self.shape.1 * 4) as u64;
        let staging = backend.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: row_size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let mut encoder = backend.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.buffer, offset, &staging, 0, row_size);
        backend.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        backend.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let flat: &[f32] = bytemuck::cast_slice(&data);
        let res = flat.to_vec();
        drop(data); staging.unmap();
        res
    }
}

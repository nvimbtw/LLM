use std::ffi::c_void;

unsafe extern "C" {
    fn cuda_malloc(ptr: *mut *mut c_void, size: usize);
    fn cuda_free(ptr: *mut c_void);
    fn cuda_memcpy_h2d(dst: *mut c_void, src: *const c_void, size: usize);
    fn cuda_memcpy_d2h(dst: *mut c_void, src: *const c_void, size: usize);
    fn cudaMemset(ptr: *mut c_void, value: i32, count: usize) -> i32;
}

pub struct DeviceBuffer {
    pub ptr: *mut f32,
    pub len: usize,
}

impl DeviceBuffer {
    pub fn alloc(len: usize) -> Self {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            cuda_malloc(&mut ptr, len * size_of::<f32>());
        }
        Self {
            ptr: ptr as *mut f32,
            len,
        }
    }

    pub fn alloc_zeroed(len: usize) -> Self {
        let buf = Self::alloc(len);
        unsafe {
            cudaMemset(buf.ptr as *mut c_void, 0, len * size_of::<f32>());
        }
        buf
    }

    pub fn from_slice(data: &[f32]) -> Self {
        let buf = Self::alloc(data.len());
        unsafe {
            cuda_memcpy_h2d(
                buf.ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                data.len() * size_of::<f32>(),
            );
        }
        buf
    }

    pub fn to_vec(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.len];
        unsafe {
            cuda_memcpy_d2h(
                out.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                self.len * size_of::<f32>(),
            );
        }
        out
    }

}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe {
            cuda_free(self.ptr as *mut c_void);
        }
    }
}

// ─── f16 GPU buffer ─────────────────────────────────────────────────────────

pub struct DeviceBufferF16 {
    pub ptr: *mut half::f16,
    pub len: usize,
}

impl DeviceBufferF16 {
    pub fn alloc(len: usize) -> Self {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        unsafe {
            cuda_malloc(&mut ptr, len * size_of::<half::f16>());
        }
        Self {
            ptr: ptr as *mut half::f16,
            len,
        }
    }

    pub fn from_f32_slice(data: &[f32]) -> Self {
        let buf = Self::alloc(data.len());
        let f16: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
        unsafe {
            cuda_memcpy_h2d(
                buf.ptr as *mut c_void,
                f16.as_ptr() as *const c_void,
                f16.len() * size_of::<half::f16>(),
            );
        }
        buf
    }

    pub fn upload_f32_slice(&self, data: &[f32]) {
        let f16: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
        unsafe {
            cuda_memcpy_h2d(
                self.ptr as *mut c_void,
                f16.as_ptr() as *const c_void,
                f16.len() * size_of::<half::f16>(),
            );
        }
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        let mut f16 = vec![half::f16::ZERO; self.len];
        unsafe {
            cuda_memcpy_d2h(
                f16.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                self.len * size_of::<half::f16>(),
            );
        }
        f16.iter().map(|&v| v.to_f32()).collect()
    }
}

impl Drop for DeviceBufferF16 {
    fn drop(&mut self) {
        unsafe {
            cuda_free(self.ptr as *mut c_void);
        }
    }
}

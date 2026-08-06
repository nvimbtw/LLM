use std::ffi::c_void;

unsafe extern "C" {
    fn cuda_malloc(ptr: *mut *mut c_void, size: usize);
    fn cuda_free(ptr: *mut c_void);
    fn cuda_memcpy_h2d(dst: *mut c_void, src: *const c_void, size: usize);
    fn cuda_memcpy_d2h(dst: *mut c_void, src: *const c_void, size: usize);
    fn cudaMemset(ptr: *mut c_void, value: i32, count: usize) -> i32;
}

pub struct Buffer {
    pub ptr: *mut f32,
    pub len: usize,
}

impl Buffer {
    pub fn alloc(len: usize) -> Self {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid, aligned out-pointer to a local `*mut c_void`;
        // cuda_malloc writes an allocation of `len * size_of::<f32>()` bytes into
        // it (or leaves it null on failure — checked implicitly by every later
        // access being sized by `len`, which matches this allocation).
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
        // SAFETY: `buf.ptr` was just allocated by `cuda_malloc` for exactly
        // `len * size_of::<f32>()` bytes, matching the count passed here.
        unsafe {
            cudaMemset(buf.ptr as *mut c_void, 0, len * size_of::<f32>());
        }
        buf
    }

    pub fn from_slice(data: &[f32]) -> Self {
        let buf = Self::alloc(data.len());
        // SAFETY: `buf.ptr` is a device allocation of `data.len() * size_of::<f32>()`
        // bytes; `data.as_ptr()` is valid for reads of that same size since it comes
        // from a live `&[f32]` slice of `data.len()` elements.
        unsafe {
            cuda_memcpy_h2d(
                buf.ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                data.len() * size_of::<f32>(),
            );
        }
        buf
    }

    /// Copy `data` into this already-allocated buffer (host→device). `data.len()`
    /// must equal `self.len` — used to seed the f32 master weights in place.
    pub fn upload(&self, data: &[f32]) {
        assert_eq!(data.len(), self.len, "upload length mismatch");
        // SAFETY: `self.ptr` is a device allocation of `self.len` f32s; `data` is a
        // live slice of the same length, so both spans are valid for the copy.
        unsafe {
            cuda_memcpy_h2d(
                self.ptr as *mut c_void,
                data.as_ptr() as *const c_void,
                data.len() * size_of::<f32>(),
            );
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.len];
        // SAFETY: `out` is a freshly allocated `Vec<f32>` of `self.len` elements,
        // so its pointer is valid for writes of `self.len * size_of::<f32>()`
        // bytes; `self.ptr` is a device allocation of the same size (invariant
        // maintained by `alloc`/`alloc_zeroed`/`from_slice`, the only ways to
        // construct a `Buffer`).
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

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was allocated by `cuda_malloc` in one of the
        // constructors and is freed exactly once here, since `Drop::drop` runs
        // at most once per value.
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
        // SAFETY: `ptr` is a valid, aligned out-pointer to a local `*mut c_void`;
        // cuda_malloc writes an allocation of `len * size_of::<half::f16>()`
        // bytes into it, matching every later access sized by `len`.
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
        // SAFETY: `buf.ptr` is a device allocation of `data.len() * size_of::<half::f16>()`
        // bytes (from `Self::alloc(data.len())`); `f16` is a live `Vec` of the same
        // length, so `f16.as_ptr()` is valid for reads of the size copied.
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
        // SAFETY: caller guarantees `self.ptr` was allocated with room for at
        // least `data.len()` f16 elements (matches `self.len` by construction
        // convention in this module); `f16` is a live `Vec` of `data.len()`
        // elements, valid for reads of the size copied.
        unsafe {
            cuda_memcpy_h2d(
                self.ptr as *mut c_void,
                f16.as_ptr() as *const c_void,
                f16.len() * size_of::<half::f16>(),
            );
        }
    }

    /// Copy back only the first `n` elements.
    ///
    /// Exists because the inference path allocates its logit buffer at
    /// `[max_s, vocab]` but usually wants one row of it: generation reads the last
    /// position only, and copying the whole allocation back per token would be
    /// megabytes of D2H and f16→f32 conversion for a few kilobytes of logits —
    /// which is the cost that made the old path slow in the first place.
    pub fn to_f32_vec_n(&self, n: usize) -> Vec<f32> {
        assert!(
            n <= self.len,
            "read of {n} elements from a buffer holding {}",
            self.len
        );
        let mut f16 = vec![half::f16::ZERO; n];
        // SAFETY: `f16` is a freshly allocated `Vec<half::f16>` of `n` elements,
        // valid for writes of `n * size_of::<half::f16>()` bytes; `self.ptr` is a
        // device allocation of `self.len >= n` elements of the same type, so the
        // source span is in bounds.
        unsafe {
            cuda_memcpy_d2h(
                f16.as_mut_ptr() as *mut c_void,
                self.ptr as *const c_void,
                n * size_of::<half::f16>(),
            );
        }
        f16.iter().map(|&v| v.to_f32()).collect()
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        let mut f16 = vec![half::f16::ZERO; self.len];
        // SAFETY: `f16` is a freshly allocated `Vec<half::f16>` of `self.len`
        // elements, valid for writes of `self.len * size_of::<half::f16>()`
        // bytes; `self.ptr` is a device allocation of the same size (invariant
        // maintained by `alloc`/`from_f32_slice`, the only ways to construct a
        // `DeviceBufferF16`).
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
        // SAFETY: `self.ptr` was allocated by `cuda_malloc` in one of the
        // constructors and is freed exactly once here, since `Drop::drop` runs
        // at most once per value.
        unsafe {
            cuda_free(self.ptr as *mut c_void);
        }
    }
}

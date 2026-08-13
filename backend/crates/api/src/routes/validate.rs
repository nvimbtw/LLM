// Boundary limits for user-supplied fields, enforced in the route layer before
// values reach the db and auth layers. Oversized input is rejected with 400 rather
// than exhausting resources (e.g. hashing a megabyte-long password) or tripping
// column limits deeper in the stack. Lengths are counted in Unicode scalar values
// (`chars()`), not bytes, so a limit means the same thing to the user regardless
// of encoding.

pub const MIN_USERNAME_LEN: usize = 1;
pub const MAX_USERNAME_LEN: usize = 64;
pub const MIN_PASSWORD_LEN: usize = 8;
pub const MAX_PASSWORD_LEN: usize = 256;
pub const MAX_MESSAGE_LEN: usize = 32_768;
pub const MAX_MODEL_NAME_LEN: usize = 128;
pub const MAX_MODEL_DESC_LEN: usize = 2_048;
// Per-file upload cap. Uploads stream to disk, so this bounds a single file's
// footprint; the multipart handler aborts and removes the partial file if hit.
pub const MAX_UPLOAD_BYTES: u64 = 25 * 1024 * 1024;

// Validates `value`'s length is within [min, max] (in characters). Returns a
// client-safe message (field name + bound, no internals) on failure.
pub fn field_len(field: &str, value: &str, min: usize, max: usize) -> Result<(), String> {
    let len = value.chars().count();
    if len < min {
        Err(format!("{} must be at least {} character(s)", field, min))
    } else if len > max {
        Err(format!("{} must be at most {} character(s)", field, max))
    } else {
        Ok(())
    }
}

// Upper-bound-only check, for fields where a lower bound is meaningless or is
// enforced elsewhere (e.g. login, which must not reveal *why* input was rejected).
pub fn max_len(field: &str, value: &str, max: usize) -> Result<(), String> {
    field_len(field, value, 0, max)
}

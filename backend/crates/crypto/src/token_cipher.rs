use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, AeadCore, KeyInit, OsRng},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use sha2::{Digest, Sha256};

// AES-GCM uses a 96-bit nonce; we store it alongside the ciphertext.
const NONCE_LEN: usize = 12;

// Build the AES-256 cipher from the JWT secret. We hash the secret (with a
// domain tag appended) rather than using it raw: SHA-256 gives the exact
// 32-byte key length AES-256 needs, and the tag keeps this key distinct from
// anything else that might be derived from the same secret.
fn build_cipher(secret: &str) -> Aes256Gcm {
    let mut hasher = Sha256::new();
    hasher.update(secret.as_bytes());
    hasher.update(b"token-enc");
    let key = hasher.finalize();
    Aes256Gcm::new_from_slice(&key).expect("sha256 digest is always 32 bytes")
}

// Encrypts a token into a base64 string of `nonce || ciphertext`. A fresh
// random nonce per call is what keeps GCM safe when the same key encrypts many
// tokens — reusing a nonce would be catastrophic, so we never hard-code it.
pub fn encrypt(plaintext: &str, secret: &str) -> Result<String, aes_gcm::Error> {
    let cipher = build_cipher(secret);
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
    let ciphertext = cipher.encrypt(&nonce, plaintext.as_bytes())?;

    let mut blob = nonce.to_vec();
    blob.extend_from_slice(&ciphertext);
    Ok(STANDARD.encode(blob))
}

// Reverses `encrypt`. Returns None on *any* failure — bad base64, a truncated
// blob, the wrong key, tampered ciphertext (GCM's auth tag catches this), or
// non-UTF8 plaintext. The caller treats them all identically: invalid token.
pub fn decrypt(token: &str, secret: &str) -> Option<String> {
    let blob = STANDARD.decode(token).ok()?;
    if blob.len() < NONCE_LEN {
        return None;
    }

    let (nonce_bytes, ciphertext) = blob.split_at(NONCE_LEN);
    let nonce = Nonce::from_slice(nonce_bytes);

    let plaintext = build_cipher(secret).decrypt(nonce, ciphertext).ok()?;
    String::from_utf8(plaintext).ok()
}

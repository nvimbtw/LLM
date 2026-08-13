use aes_gcm::{
    Aes256Gcm, Nonce,
    aead::{Aead, AeadCore, KeyInit, OsRng},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use sha2::{Digest, Sha256};

/// AES-GCM's nonce, stored alongside the ciphertext.
const NONCE_LEN: usize = 12;

/// The secret is hashed rather than used raw: SHA-256 gives the 32 bytes
/// AES-256 needs, and the domain tag keeps this key distinct from anything else
/// derived from the same secret.
fn build_cipher(secret: &str) -> Aes256Gcm {
    let mut hasher = Sha256::new();
    hasher.update(secret.as_bytes());
    hasher.update(b"token-enc");
    let key = hasher.finalize();
    Aes256Gcm::new_from_slice(&key).expect("sha256 digest is always 32 bytes")
}

/// Base64 of `nonce || ciphertext`. The nonce is fresh per call, which is what
/// keeps GCM safe when one key encrypts many tokens.
pub fn encrypt(plaintext: &str, secret: &str) -> Result<String, aes_gcm::Error> {
    let cipher = build_cipher(secret);
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
    let ciphertext = cipher.encrypt(&nonce, plaintext.as_bytes())?;

    let mut blob = nonce.to_vec();
    blob.extend_from_slice(&ciphertext);
    Ok(STANDARD.encode(blob))
}

/// None on *any* failure — bad base64, truncation, the wrong key, tampering
/// (GCM's auth tag catches that), bad UTF-8. All of them mean: invalid token.
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

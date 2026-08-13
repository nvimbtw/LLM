// Shared helpers for the WebSocket routes (chat, training). Browsers can't set
// the `Authorization` header on a WebSocket handshake, so every socket
// authenticates by sending its token in the first text frame instead — this
// module decodes that frame the same way the HTTP `AuthUser` extractor does.

use serde::Deserialize;

use auth::token::{self as jwt, Claims};

// The very first frame a client sends over any socket: the encrypted token,
// the same value the HTTP routes pass as `Authorization: Bearer`.
#[derive(Deserialize)]
struct AuthFrame {
    token: String,
}

// Decrypts + verifies the token frame, yielding the caller's claims (user id +
// access level). None on any failure (bad JSON, bad/expired token) — the same
// all-or-nothing stance as the `AuthUser` extractor.
pub fn authenticate(text: &str, secret: &str) -> Option<Claims> {
    let frame: AuthFrame = serde_json::from_str(text).ok()?;
    let jwt = auth::token_cipher::decrypt(&frame.token, secret)?;
    jwt::verify_user_token(&jwt, secret).ok()
}

// Standard error frame shared by the socket handlers.
pub fn error_frame(message: &str) -> String {
    serde_json::json!({ "type": "error", "error": message }).to_string()
}

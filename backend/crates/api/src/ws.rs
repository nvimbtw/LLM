//! The frame every socket opens with.
//!
//! A browser cannot set a header on a WebSocket handshake, so the token arrives
//! as the first message instead — the same value the HTTP routes carry as
//! `Authorization: Bearer`, verified the same way.

use serde::Deserialize;

use auth::token::{self as jwt, Claims};

#[derive(Deserialize)]
struct AuthFrame {
    token: String,
}

pub fn authenticate(text: &str, secret: &str) -> Option<Claims> {
    let frame: AuthFrame = serde_json::from_str(text).ok()?;
    let jwt = auth::token_cipher::decrypt(&frame.token, secret)?;
    jwt::verify_user_token(&jwt, secret).ok()
}

pub fn error_frame(message: &str) -> String {
    serde_json::json!({ "type": "error", "error": message }).to_string()
}

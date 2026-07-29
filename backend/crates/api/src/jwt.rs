use jsonwebtoken::{DecodingKey, EncodingKey, Header, Validation, decode, encode};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use db::user::AccessLevel;

#[derive(Serialize, Deserialize)]
pub struct Claims {
    pub sub: Uuid,
    pub access_level: AccessLevel,
    pub exp: usize,
}

pub fn generate_token(
    user_id: Uuid,
    access_level: AccessLevel,
    secret: &str,
) -> Result<String, jsonwebtoken::errors::Error> {
    let expiration = chrono::Utc::now()
        .checked_add_signed(chrono::Duration::hours(24))
        .expect("valid timestamp")
        .timestamp() as usize;

    let claims = Claims {
        sub: user_id,
        access_level,
        exp: expiration,
    };

    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
}

pub fn verify_token(token: &str, secret: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::default(),
    )?;
    Ok(token_data.claims)
}

pub mod benchmark;
pub mod chat;
pub mod cycle;
pub mod job;
pub mod message;
pub mod model;
pub mod preset;
pub mod signal;
pub mod usage;
pub mod user;

use crate::crypto::password;
use sqlx::postgres::{PgPool, PgPoolOptions};

pub async fn create_pool(database_url: &str) -> Result<PgPool, sqlx::Error> {
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(database_url)
        .await?;

    sqlx::migrate!("./src/db/migrations").run(&pool).await?;

    let pwd_hash = match password::hash_password("admin") {
        Ok(hash) => hash,
        Err(e) => panic!("Failed to hash password {}", e),
    };

    match user::create(
        &pool,
        "admin",
        pwd_hash.as_str(),
        Some(user::AccessLevel::Admin),
    )
    .await
    {
        Ok(_) => println!("Created admin!"),
        Err(_) => println!("Admin already exists"),
    }

    Ok(pool)
}

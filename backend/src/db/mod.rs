pub mod chat;
pub mod message;
pub mod model;
pub mod user;

use sqlx::postgres::{PgPool, PgPoolOptions};

pub async fn create_pool(database_url: &str) -> Result<PgPool, sqlx::Error> {
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(database_url)
        .await?;

    sqlx::migrate!("./src/db/migrations").run(&pool).await?;

    Ok(pool)
}

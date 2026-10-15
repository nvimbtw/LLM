pub mod attachment;
pub mod conversation;
pub mod preset;
pub mod project;
pub mod user;

#[cfg(test)]
mod tests;

use auth::password;
use sqlx::postgres::{PgPool, PgPoolOptions};

/// Connect to the supplied database, apply migrations, and seed missing login accounts.
/// Returns the pool or a connection, migration, password-hashing, or insertion error.
pub async fn create_pool(database_url: &str) -> Result<PgPool, sqlx::Error> {
    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(database_url)
        .await?;

    sqlx::migrate!("./migrations").run(&pool).await?;
    insert_users(&pool).await?;

    Ok(pool)
}

/// Seed the admin and agent accounts in the migrated pool without changing existing users.
/// Concurrent inserts of the same username are harmless; hashing and other database failures propagate.
async fn insert_users(pool: &PgPool) -> Result<(), sqlx::Error> {
    for (username, plaintext_password) in [("admin", "admin"), ("agents", "agents")] {
        let password_hash = password::hash_password(plaintext_password).map_err(|error| {
            sqlx::Error::Configuration(
                format!("failed to hash bootstrap password for {username}: {error}").into(),
            )
        })?;
        match user::create(
            pool,
            username,
            &password_hash,
            Some(user::AccessLevel::Admin),
        )
        .await
        {
            Ok(_) => println!("Created {username}!"),
            Err(sqlx::Error::Database(error))
                if error.is_unique_violation()
                    && error.constraint() == Some("users_username_key") => {}
            Err(error) => return Err(error),
        }
    }

    Ok(())
}

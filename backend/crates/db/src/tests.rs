use super::{create_pool, insert_users, user};
use auth::password;
use sqlx::{ConnectOptions, PgPool};

/// Startup migrates an empty database and creates distinct admin logins with usable password hashes.
#[sqlx::test(migrations = false)]
async fn startup_seeds_login_accounts(pool: PgPool) -> Result<(), sqlx::Error> {
    let database_url = pool.connect_options().to_url_lossy();
    let started = create_pool(database_url.as_str()).await?;
    let admin = user::find_by_username(&started, "admin").await?.unwrap();
    let agents = user::find_by_username(&started, "agents").await?.unwrap();

    assert_ne!(admin.id_user, agents.id_user);
    for account in [admin, agents] {
        assert_eq!(account.access_level, user::AccessLevel::Admin);
        assert_ne!(account.password, account.username);
        assert!(password::verify_password(&account.username, &account.password).unwrap());
    }
    started.close().await;
    Ok(())
}

/// Restarts add missing defaults while preserving existing user IDs, passwords and access levels.
#[sqlx::test(migrations = "./migrations")]
async fn startup_preserves_existing_accounts(pool: PgPool) -> Result<(), sqlx::Error> {
    let saved_password = password::hash_password("saved-password").unwrap();
    let agents = user::create(
        &pool,
        "agents",
        &saved_password,
        Some(user::AccessLevel::User),
    )
    .await?;
    let database_url = pool.connect_options().to_url_lossy();
    let first = create_pool(database_url.as_str()).await?;
    let admin = user::find_by_username(&first, "admin").await?.unwrap();
    user::update(&first, admin.id_user, None, Some(&saved_password)).await?;
    first.close().await;

    let restarted = create_pool(database_url.as_str()).await?;
    for (original, access_level) in [
        (admin, user::AccessLevel::Admin),
        (agents, user::AccessLevel::User),
    ] {
        let saved = user::find_by_username(&restarted, &original.username)
            .await?
            .unwrap();
        assert_eq!(saved.id_user, original.id_user);
        assert_eq!(saved.password, saved_password);
        assert_eq!(saved.access_level, access_level);
    }
    restarted.close().await;
    Ok(())
}

/// Unmigrated storage fails seeding instead of being mistaken for already populated accounts.
#[sqlx::test(migrations = false)]
async fn bootstrap_propagates_database_failures(pool: PgPool) {
    let error = insert_users(&pool).await.unwrap_err();
    assert!(!error.as_database_error().unwrap().is_unique_violation());
}

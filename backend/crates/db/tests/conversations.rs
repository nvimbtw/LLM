//! Ownership and persistence contracts transferred from the Python service.
use db::{attachment, conversation as c, user};
use sqlx::PgPool;
use uuid::Uuid;

fn message(request: Uuid, role: &str) -> c::NewMessage {
    c::NewMessage {
        role: role.into(),
        content: "exact text 🦀".into(),
        tokens: 7,
        model_id: None,
        model_name: Some("original name".into()),
        tool: None,
        argument: None,
        request_id: request,
    }
}

#[sqlx::test(migrations = "./migrations")]
async fn owners_bytes_binding_and_cascades(pool: PgPool) -> Result<(), sqlx::Error> {
    let alice = user::create(&pool, "alice", "hash", None).await?.id_user;
    let bob = user::create(&pool, "bob", "hash", None).await?.id_user;
    let chat = c::create(&pool, alice, "private").await?;
    let bytes = b"\0\xff\x89PNG\r\n";
    let file = attachment::create(&pool, alice, "photo.png", "image/png", bytes).await?;
    assert!(c::get(&pool, bob, chat.id).await?.is_none());
    assert!(attachment::get(&pool, bob, file.id).await?.is_none());
    assert!(!attachment::discard(&pool, bob, file.id).await?);
    assert_eq!(
        attachment::get(&pool, alice, file.id).await?.unwrap().1,
        bytes
    );
    let stranger = attachment::create(&pool, bob, "b.txt", "text/plain", b"b").await?;
    assert!(c::append(
        &pool,
        alice,
        chat.id,
        &message(Uuid::new_v4(), "user"),
        &[file.id, stranger.id]
    )
    .await
    .is_err());
    assert!(c::messages(&pool, alice, chat.id).await?.is_empty());
    let request = Uuid::new_v4();
    let input = message(request, "user");
    let sent = c::append(&pool, alice, chat.id, &input, &[file.id]).await?;
    assert!(!attachment::discard(&pool, alice, file.id).await?);
    assert!(c::append(&pool, alice, chat.id, &input, &[]).await.is_err());
    assert_eq!(c::messages(&pool, alice, chat.id).await?.len(), 1);
    assert!(!c::feedback(&pool, bob, sent.id, Some("up")).await?);
    assert!(c::feedback(&pool, alice, sent.id, Some("up")).await?);
    assert_eq!(
        c::messages(&pool, alice, chat.id).await?[0]
            .feedback
            .as_deref(),
        Some("up")
    );
    assert!(!c::remove(&pool, bob, chat.id).await?);
    assert!(c::remove(&pool, alice, chat.id).await?);
    assert!(attachment::get(&pool, alice, file.id).await?.is_none());
    assert!(attachment::get(&pool, bob, stranger.id).await?.is_some());
    Ok(())
}

#[sqlx::test(migrations = "./migrations")]
async fn ordered_turns_usage_and_attribution(pool: PgPool) -> Result<(), sqlx::Error> {
    let owner = user::create(&pool, "owner", "hash", None).await?.id_user;
    let chat = c::create(&pool, owner, "c").await?;
    let request = Uuid::new_v4();
    c::append_many(
        &pool,
        owner,
        chat.id,
        &[message(request, "tool"), message(request, "user")],
        &[],
    )
    .await?;
    let mut reply = message(request, "assistant");
    reply.model_id = Some(Uuid::new_v4());
    c::append(&pool, owner, chat.id, &reply, &[]).await?;
    let rows = c::messages(&pool, owner, chat.id).await?;
    assert_eq!(
        rows.iter().map(|row| row.role.as_str()).collect::<Vec<_>>(),
        vec!["tool", "user", "assistant"]
    );
    assert_eq!(rows[2].model_name.as_deref(), Some("original name"));
    let usage = c::usage(&pool, owner, 30).await?;
    assert_eq!(usage["series"][0]["asked"], 7);
    assert_eq!(usage["series"][0]["generated"], 7);
    assert_eq!(usage["by_model"][0]["name"], "original name");
    assert_eq!(c::get(&pool, owner, chat.id).await?.unwrap().tokens, 21);
    user::delete(&pool, owner).await?;
    assert!(c::get(&pool, owner, chat.id).await?.is_none());
    Ok(())
}

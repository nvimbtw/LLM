//! Real PostgreSQL contracts for private project membership and indexed message retrieval.

use db::{conversation as chats, project, user};
use sqlx::PgPool;
use uuid::Uuid;

/// Build a stored message with an independent request identity and a fixed token count.
fn message(role: &str, content: &str) -> chats::NewMessage {
    chats::NewMessage {
        role: role.into(),
        content: content.into(),
        tokens: 4,
        model_id: None,
        model_name: None,
        tool: None,
        argument: None,
        request_id: Uuid::new_v4(),
    }
}

/// Wait at most two seconds for the detail query's message-table read to reach the deliberately held lock.
async fn wait_for_message_read(pool: &PgPool) {
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            let waiting = sqlx::query_scalar!(
                r#"SELECT count(*) AS "count!" FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid
                   WHERE l.relation = 'messages'::regclass AND NOT l.granted AND a.datname = current_database()"#
            ).fetch_one(pool).await.unwrap();
            if waiting > 0 {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }).await.expect("detail query reached its thread-page read");
}

/// A chat committed between the metadata and thread reads cannot enter a page whose snapshot counted no chats.
#[sqlx::test(migrations = "./migrations")]
async fn detail_count_and_page_share_a_snapshot(pool: PgPool) -> Result<(), sqlx::Error> {
    let owner = user::create(&pool, "owner", "hash", None).await?.id_user;
    let stranger = user::create(&pool, "stranger", "hash", None).await?.id_user;
    let scope = project::create(&pool, owner, "scope", "", "").await?;
    let mut gate = pool.begin().await?;
    sqlx::query!("LOCK TABLE messages IN ACCESS EXCLUSIVE MODE")
        .execute(&mut *gate)
        .await?;
    let reader_pool = pool.clone();
    let pending =
        tokio::spawn(async move { project::detail(&reader_pool, owner, scope.id, 50, 0).await });
    wait_for_message_read(&pool).await;
    let created =
        chats::create_in_project(&pool, owner, "added concurrently", Some(scope.id)).await?;
    gate.commit().await?;
    let before = pending.await.unwrap()?.unwrap();
    assert_eq!(before.project.thread_count, 0);
    assert!(before.threads.is_empty());
    let after = project::detail(&pool, owner, scope.id, 50, 0)
        .await?
        .unwrap();
    assert_eq!(after.project.thread_count, 1);
    assert_eq!(after.threads[0].id, created.id);
    let beyond = project::detail(&pool, owner, scope.id, 1, 1)
        .await?
        .unwrap();
    assert_eq!(beyond.project.thread_count, 1);
    assert!(beyond.threads.is_empty());
    assert!(project::detail(&pool, stranger, scope.id, 50, 0)
        .await?
        .is_none());
    Ok(())
}

/// Verify owner checks, atomic chat assignment, direct DB ownership enforcement and deletion preservation.
#[sqlx::test(migrations = "./migrations")]
async fn membership_is_private_and_deletion_keeps_chats(pool: PgPool) -> Result<(), sqlx::Error> {
    let alice = user::create(&pool, "alice", "hash", None).await?.id_user;
    let bob = user::create(&pool, "bob", "hash", None).await?.id_user;
    let first = project::create(&pool, alice, "first", "description", "notes").await?;
    let second = project::create(&pool, alice, "second", "", "").await?;
    let foreign = project::create(&pool, bob, "foreign", "", "").await?;
    assert!(project::get(&pool, bob, first.id).await?.is_none());
    assert!(!project::update(&pool, bob, first.id, Some("stolen"), None, None).await?);
    assert!(!project::remove(&pool, bob, first.id).await?);
    assert!(
        chats::create_in_project(&pool, alice, "invalid", Some(foreign.id))
            .await
            .is_err()
    );
    assert!(chats::list(&pool, alice).await?.is_empty());
    let chat = chats::create_in_project(&pool, alice, "source", Some(first.id)).await?;
    let stored =
        chats::append(&pool, alice, chat.id, &message("user", "orbit design"), &[]).await?;
    assert!(chats::assign_project(&pool, bob, chat.id, None)
        .await?
        .is_none());
    assert!(
        chats::assign_project(&pool, alice, chat.id, Some(foreign.id))
            .await?
            .is_none()
    );
    assert!(sqlx::query!(
        "UPDATE chats SET project_id = $1 WHERE id = $2",
        foreign.id,
        chat.id
    )
    .execute(&pool)
    .await
    .is_err());
    assert_eq!(
        chats::get(&pool, alice, chat.id).await?.unwrap().project_id,
        Some(first.id)
    );
    let moved = chats::assign_project(&pool, alice, chat.id, Some(second.id))
        .await?
        .unwrap();
    assert_eq!(moved.project_id, Some(second.id));
    assert_eq!(
        project::get(&pool, alice, first.id)
            .await?
            .unwrap()
            .thread_count,
        0
    );
    assert_eq!(
        project::get(&pool, alice, second.id)
            .await?
            .unwrap()
            .thread_count,
        1
    );
    assert!(project::remove(&pool, alice, second.id).await?);
    assert_eq!(
        chats::get(&pool, alice, chat.id).await?.unwrap().project_id,
        None
    );
    assert_eq!(
        chats::messages(&pool, alice, chat.id).await?[0].id,
        stored.id
    );
    assert!(project::context(&pool, alice, chat.id, Some("orbit"))
        .await?
        .is_none());
    assert_eq!(project::list(&pool, alice, 1, 0).await?.1, 1);
    assert!(project::list(&pool, alice, 1, 10).await?.0.is_empty());
    Ok(())
}

/// A recent slice preserves account isolation, newest-first ordering and persisted token totals.
#[sqlx::test(migrations = "./migrations")]
async fn recent_chat_slice_matches_archive(pool: PgPool) -> Result<(), sqlx::Error> {
    let owner = user::create(&pool, "recent-owner", "hash", None)
        .await?
        .id_user;
    let stranger = user::create(&pool, "recent-stranger", "hash", None)
        .await?
        .id_user;
    let first = chats::create(&pool, owner, "first").await?;
    chats::append(
        &pool,
        owner,
        first.id,
        &message("user", "first message"),
        &[],
    )
    .await?;
    let second = chats::create(&pool, owner, "second").await?;
    chats::append(
        &pool,
        owner,
        second.id,
        &message("user", "second message"),
        &[],
    )
    .await?;
    chats::append(&pool, owner, second.id, &message("assistant", "reply"), &[]).await?;
    chats::create(&pool, stranger, "private").await?;
    let archive = chats::list(&pool, owner).await?;
    let recent = chats::list_recent(&pool, owner, 1).await?;
    assert_eq!(archive.len(), 2);
    assert_eq!(recent.len(), 1);
    assert_eq!(recent[0].id, second.id);
    assert_eq!(recent[0].tokens, 8);
    assert!(chats::list_recent(&pool, stranger, 1)
        .await?
        .iter()
        .all(|chat| chat.id != second.id));
    Ok(())
}

/// Matching accepts a shared query word and keeps only user/assistant sources from other owned project threads.
#[sqlx::test(migrations = "./migrations")]
async fn memories_keep_real_sources_and_project_boundaries(
    pool: PgPool,
) -> Result<(), sqlx::Error> {
    let alice = user::create(&pool, "alice", "hash", None).await?.id_user;
    let bob = user::create(&pool, "bob", "hash", None).await?.id_user;
    let scope = project::create(&pool, alice, "scope", "", "standing notes").await?;
    let other = project::create(&pool, alice, "other", "", "").await?;
    let foreign = project::create(&pool, bob, "foreign", "", "").await?;
    let current = chats::create_in_project(&pool, alice, "current", Some(scope.id)).await?;
    let source = chats::create_in_project(&pool, alice, "source", Some(scope.id)).await?;
    let unrelated = chats::create_in_project(&pool, alice, "unrelated", Some(other.id)).await?;
    let stranger = chats::create_in_project(&pool, bob, "stranger", Some(foreign.id)).await?;
    let user_source = chats::append(
        &pool,
        alice,
        source.id,
        &message("user", "orbit battery plan"),
        &[],
    )
    .await?;
    let reply_source = chats::append(
        &pool,
        alice,
        source.id,
        &message("assistant", "orbit reply"),
        &[],
    )
    .await?;
    chats::append(&pool, alice, source.id, &message("tool", "orbit tool"), &[]).await?;
    chats::append(
        &pool,
        alice,
        current.id,
        &message("user", "orbit current"),
        &[],
    )
    .await?;
    chats::append(
        &pool,
        alice,
        unrelated.id,
        &message("user", "orbit unrelated"),
        &[],
    )
    .await?;
    chats::append(
        &pool,
        bob,
        stranger.id,
        &message("user", "orbit private"),
        &[],
    )
    .await?;
    let context = project::context(&pool, alice, current.id, Some("orbit | schedule"))
        .await?
        .unwrap();
    assert_eq!(context.project.notes, "standing notes");
    assert_eq!(context.memories.len(), 2);
    assert!(context
        .memories
        .iter()
        .all(|memory| memory.thread == source.id));
    assert!(context
        .memories
        .iter()
        .any(|memory| memory.message == user_source.id && memory.role == "user"));
    assert!(context
        .memories
        .iter()
        .any(|memory| memory.message == reply_source.id && memory.role == "assistant"));
    assert!(context
        .memories
        .iter()
        .all(|memory| memory.age_seconds >= 0));
    assert!(project::context(&pool, bob, current.id, Some("orbit"))
        .await?
        .is_none());
    assert!(project::context(&pool, alice, current.id, None)
        .await?
        .unwrap()
        .memories
        .is_empty());
    chats::assign_project(&pool, alice, source.id, Some(other.id)).await?;
    assert!(project::context(&pool, alice, current.id, Some("orbit"))
        .await?
        .unwrap()
        .memories
        .is_empty());
    Ok(())
}

/// Retrieval caps count and Unicode snippet length, excludes nonmatches and clamps future timestamps to zero age.
#[sqlx::test(migrations = "./migrations")]
async fn memory_budgets_and_age_are_enforced_in_storage(pool: PgPool) -> Result<(), sqlx::Error> {
    let owner = user::create(&pool, "owner", "hash", None).await?.id_user;
    let scope = project::create(&pool, owner, "scope", "", "").await?;
    let current = chats::create_in_project(&pool, owner, "current", Some(scope.id)).await?;
    let source = chats::create_in_project(&pool, owner, "source", Some(scope.id)).await?;
    for _ in 0..12 {
        chats::append(
            &pool,
            owner,
            source.id,
            &message("user", &format!("orbit {}", "🦀".repeat(2200))),
            &[],
        )
        .await?;
    }
    let future = chats::append(
        &pool,
        owner,
        source.id,
        &message("assistant", "orbit future"),
        &[],
    )
    .await?;
    sqlx::query!(
        "UPDATE messages SET created_at = now() + interval '1 day' WHERE id = $1",
        future.id
    )
    .execute(&pool)
    .await?;
    let context = project::context(&pool, owner, current.id, Some("orbit"))
        .await?
        .unwrap();
    assert_eq!(context.memories.len(), 8);
    assert!(context
        .memories
        .iter()
        .all(|memory| memory.memory.chars().count() <= 2000));
    assert!(context
        .memories
        .iter()
        .any(|memory| memory.message == future.id && memory.age_seconds == 0));
    assert!(
        project::context(&pool, owner, current.id, Some("unmatched"))
            .await?
            .unwrap()
            .memories
            .is_empty()
    );
    assert!(project::create(&pool, owner, " ", "", "").await.is_err());
    assert!(
        project::create(&pool, owner, "scope", &"x".repeat(2001), "")
            .await
            .is_err()
    );
    assert!(
        project::create(&pool, owner, "scope", "", &"x".repeat(8001))
            .await
            .is_err()
    );
    Ok(())
}

/// A shared word after long leading prose appears in the retrieved excerpt, while words beyond the scan ceiling do not match.
#[sqlx::test(migrations = "./migrations")]
async fn matched_excerpt_reaches_late_words_with_a_bounded_scan(
    pool: PgPool,
) -> Result<(), sqlx::Error> {
    let owner = user::create(&pool, "owner", "hash", None).await?.id_user;
    let scope = project::create(&pool, owner, "scope", "", "").await?;
    let current = chats::create_in_project(&pool, owner, "current", Some(scope.id)).await?;
    let source = chats::create_in_project(&pool, owner, "source", Some(scope.id)).await?;
    let stored = chats::append(
        &pool,
        owner,
        source.id,
        &message(
            "assistant",
            &format!(
                "{} orbitalbattery finishes Friday",
                "old unrelated prose ".repeat(250)
            ),
        ),
        &[],
    )
    .await?;
    let context = project::context(&pool, owner, current.id, Some("orbitalbattery"))
        .await?
        .unwrap();
    assert_eq!(context.memories.len(), 1);
    assert_eq!(context.memories[0].message, stored.id);
    assert!(context.memories[0].memory.contains("orbitalbattery"));
    assert!(context.memories[0].memory.chars().count() <= 2000);
    chats::append(
        &pool,
        owner,
        source.id,
        &message(
            "user",
            &format!("{} invisiblekeyword", "unrelated ".repeat(1000)),
        ),
        &[],
    )
    .await?;
    assert!(
        project::context(&pool, owner, current.id, Some("invisiblekeyword"))
            .await?
            .unwrap()
            .memories
            .is_empty()
    );
    Ok(())
}

/// The additive feature migration handles a very large preexisting message without building an unbounded word vector.
#[sqlx::test(migrations = false)]
async fn migration_bounds_indexing_for_legacy_messages(pool: PgPool) -> Result<(), sqlx::Error> {
    let full = sqlx::migrate!("./migrations");
    let mut legacy = sqlx::migrate::Migrator::DEFAULT;
    legacy.migrations = full
        .iter()
        .filter(|migration| migration.version < 20261014000000)
        .cloned()
        .collect::<Vec<_>>()
        .into();
    legacy.run(&pool).await?;
    let owner = user::create(&pool, "owner", "hash", None).await?.id_user;
    let id = Uuid::new_v4();
    sqlx::query!(
        "INSERT INTO chats(id, owner_id, title) VALUES ($1, $2, 'legacy')",
        id,
        owner
    )
    .execute(&pool)
    .await?;
    let large = (0..150000)
        .map(|word| format!("legacyword{word}"))
        .collect::<Vec<_>>()
        .join(" ");
    let request = Uuid::new_v4();
    sqlx::query!("INSERT INTO messages(chat_id, position, role, content, request_id) VALUES ($1, 1, 'user', $2, $3)",
        id, large, request).execute(&pool).await?;
    full.run(&pool).await?;
    let chat = chats::get(&pool, owner, id).await?.unwrap();
    assert_eq!(chat.project_id, None);
    let count = sqlx::query_scalar!(
        r#"SELECT length(memory_words) AS "count!" FROM messages WHERE chat_id = $1"#,
        id
    )
    .fetch_one(&pool)
    .await?;
    assert!(count < 1000);
    assert_eq!(chats::messages(&pool, owner, id).await?[0].content, large);
    Ok(())
}

/// Hidden complete and unfinished assistant reasoning never participates in matching or excerpts, while quoted user markers stay literal.
#[sqlx::test(migrations = "./migrations")]
async fn reasoning_is_removed_before_matching_and_excerpting(
    pool: PgPool,
) -> Result<(), sqlx::Error> {
    let owner = user::create(&pool, "owner", "hash", None).await?.id_user;
    let scope = project::create(&pool, owner, "scope", "", "").await?;
    let current = chats::create_in_project(&pool, owner, "current", Some(scope.id)).await?;
    let source = chats::create_in_project(&pool, owner, "source", Some(scope.id)).await?;
    let stored = chats::append(
        &pool,
        owner,
        source.id,
        &message(
            "assistant",
            &format!(
                "<|thinking|>hiddensecret\n{}<|/thinking|> orbitalbattery visible answer",
                "unrelated ".repeat(250)
            ),
        ),
        &[],
    )
    .await?;
    chats::append(
        &pool,
        owner,
        source.id,
        &message(
            "assistant",
            "visible start<|thinking|>unfinishedsecret orbitalbattery secret suffix",
        ),
        &[],
    )
    .await?;
    assert!(project::context(
        &pool,
        owner,
        current.id,
        Some("hiddensecret | unfinishedsecret")
    )
    .await?
    .unwrap()
    .memories
    .is_empty());
    let context = project::context(&pool, owner, current.id, Some("orbitalbattery"))
        .await?
        .unwrap();
    assert_eq!(context.memories.len(), 1);
    assert_eq!(context.memories[0].message, stored.id);
    assert!(context.memories[0]
        .memory
        .contains("orbitalbattery visible answer"));
    assert!(!context.memories[0].memory.contains("hiddensecret"));
    assert!(!context.memories[0].memory.contains("thinking"));
    let literal = chats::append(
        &pool,
        owner,
        source.id,
        &message("user", "quote <|thinking|>literalkeyword<|/thinking|>"),
        &[],
    )
    .await?;
    let context = project::context(&pool, owner, current.id, Some("literalkeyword"))
        .await?
        .unwrap();
    assert_eq!(context.memories[0].message, literal.id);
    assert_eq!(
        context.memories[0].memory,
        "quote <|thinking|>literalkeyword<|/thinking|>"
    );
    Ok(())
}

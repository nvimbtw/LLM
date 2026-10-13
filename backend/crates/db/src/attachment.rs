use chrono::{DateTime, Utc};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

#[derive(FromRow)]
pub struct Metadata {
    pub id: Uuid,
    pub message_id: Option<Uuid>,
    pub filename: String,
    pub media_type: String,
    pub size: i32,
    pub created_at: DateTime<Utc>,
}

pub async fn create(
    pool: &PgPool,
    owner: Uuid,
    filename: &str,
    media_type: &str,
    bytes: &[u8],
) -> Result<Metadata, sqlx::Error> {
    sqlx::query_as!(Metadata, r#"INSERT INTO attachments(owner_id,filename,media_type,size,content) VALUES($1,$2,$3,octet_length($4::bytea),$4) RETURNING id,message_id,filename,media_type,size,created_at"#, owner, filename, media_type, bytes).fetch_one(pool).await
}

pub async fn get(
    pool: &PgPool,
    owner: Uuid,
    id: Uuid,
) -> Result<Option<(Metadata, Vec<u8>)>, sqlx::Error> {
    let metadata: Option<Metadata> = sqlx::query_as!(Metadata, r#"SELECT id,message_id,filename,media_type,size,created_at FROM attachments WHERE id=$1 AND owner_id=$2"#, id, owner).fetch_optional(pool).await?;
    let Some(metadata) = metadata else {
        return Ok(None);
    };
    let bytes = sqlx::query_scalar!(
        r#"SELECT content FROM attachments WHERE id=$1 AND owner_id=$2"#,
        id,
        owner
    )
    .fetch_optional(pool)
    .await?;
    Ok(bytes.map(|bytes| (metadata, bytes)))
}

pub async fn for_message(
    pool: &PgPool,
    owner: Uuid,
    message: Uuid,
) -> Result<Vec<Metadata>, sqlx::Error> {
    sqlx::query_as!(Metadata, r#"SELECT id,message_id,filename,media_type,size,created_at FROM attachments WHERE message_id=$1 AND owner_id=$2 ORDER BY created_at,id"#, message, owner).fetch_all(pool).await
}

pub async fn discard(pool: &PgPool, owner: Uuid, id: Uuid) -> Result<bool, sqlx::Error> {
    Ok(sqlx::query!(
        r#"DELETE FROM attachments WHERE id=$1 AND owner_id=$2 AND message_id IS NULL"#,
        id,
        owner
    )
    .execute(pool)
    .await?
    .rows_affected()
        == 1)
}

pub async fn for_chat(
    pool: &PgPool,
    owner: Uuid,
    chat: Uuid,
) -> Result<Vec<Metadata>, sqlx::Error> {
    sqlx::query_as!(Metadata, "SELECT a.id,a.message_id,a.filename,a.media_type,a.size,a.created_at FROM attachments a JOIN messages m ON m.id=a.message_id JOIN chats c ON c.id=m.chat_id WHERE c.id=$1 AND c.owner_id=$2 AND a.owner_id=$2 ORDER BY a.created_at,a.id",chat,owner).fetch_all(pool).await
}

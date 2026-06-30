use chrono::NaiveDate;
use sqlx::PgPool;
use uuid::Uuid;

// One generation turn's token counts. Recorded after a successful chat reply.
pub async fn record(
    pool: &PgPool,
    id_user: Uuid,
    id_model: Option<Uuid>,
    input_tokens: i32,
    output_tokens: i32,
) -> Result<(), sqlx::Error> {
    sqlx::query!(
        r#"INSERT INTO usage_events (id_user, id_model, input_tokens, output_tokens)
           VALUES ($1, $2, $3, $4)"#,
        id_user,
        id_model,
        input_tokens,
        output_tokens,
    )
    .execute(pool)
    .await?;
    Ok(())
}

// One day's totals for a user. Only days with activity are returned; the route
// fills the gaps so the chart gets a contiguous series.
pub struct DailyUsage {
    pub day: NaiveDate,
    pub input: i64,
    pub output: i64,
    pub requests: i64,
}

pub async fn daily_for_user(
    pool: &PgPool,
    id_user: Uuid,
    days: i32,
) -> Result<Vec<DailyUsage>, sqlx::Error> {
    sqlx::query_as!(
        DailyUsage,
        r#"SELECT
               (created_at AT TIME ZONE 'UTC')::date AS "day!",
               COALESCE(SUM(input_tokens), 0)::bigint AS "input!",
               COALESCE(SUM(output_tokens), 0)::bigint AS "output!",
               COUNT(*)::bigint AS "requests!"
           FROM usage_events
           WHERE id_user = $1
             AND created_at >= now() - make_interval(days => $2)
           GROUP BY (created_at AT TIME ZONE 'UTC')::date
           ORDER BY (created_at AT TIME ZONE 'UTC')::date ASC"#,
        id_user,
        days,
    )
    .fetch_all(pool)
    .await
}

// Per-model totals over the window. A null id_model (model since deleted) is
// surfaced with a null name so the route can label it.
pub struct ModelUsage {
    pub id_model: Option<Uuid>,
    pub name: Option<String>,
    pub tokens: i64,
    pub requests: i64,
}

pub async fn by_model_for_user(
    pool: &PgPool,
    id_user: Uuid,
    days: i32,
) -> Result<Vec<ModelUsage>, sqlx::Error> {
    sqlx::query_as!(
        ModelUsage,
        r#"SELECT
               u.id_model AS "id_model?",
               m.name AS "name?",
               COALESCE(SUM(u.input_tokens + u.output_tokens), 0)::bigint AS "tokens!",
               COUNT(*)::bigint AS "requests!"
           FROM usage_events u
           LEFT JOIN models m ON m.id_model = u.id_model
           WHERE u.id_user = $1
             AND u.created_at >= now() - make_interval(days => $2)
           GROUP BY u.id_model, m.name
           ORDER BY SUM(u.input_tokens + u.output_tokens) DESC"#,
        id_user,
        days,
    )
    .fetch_all(pool)
    .await
}

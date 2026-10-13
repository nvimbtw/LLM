//! Usage is derived from the owner's stored messages.
use crate::upstream::{answer, Answer};
use crate::{conversation, extractors::AuthUser};
use actix_web::{get, web};
use serde::Deserialize;
use sqlx::PgPool;
#[derive(Deserialize)]
struct Window {
    days: Option<i32>,
}
#[get("")]
async fn get_usage(user: AuthUser, query: web::Query<Window>, pool: web::Data<PgPool>) -> Answer {
    let days = query.days.unwrap_or(30);
    if !(1..=365).contains(&days) {
        return Err(conversation::failure(400, "days must be between 1 and 365").into());
    }
    answer(
        db::conversation::usage(&pool, user.claims.sub, days)
            .await
            .map_err(conversation::database)?,
    )
}
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(get_usage);
}

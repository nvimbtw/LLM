//! What a person's conversations have cost, counted where the turns are stored.

use actix_web::{get, web, HttpRequest};
use llm_engine::Proxy;

use crate::extractors::AuthUser;
use crate::upstream::{answer, caller, with_query, Answer};

#[get("")]
async fn get_usage(user: AuthUser, req: HttpRequest, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .get(&caller(&user), &with_query("/usage", &req))
            .await?,
    )
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(get_usage);
}

//! What a person's conversations have cost. Counted where the turns are stored.

use actix_web::{HttpRequest, get, web};
use engine::Proxy;

use crate::extractors::AuthUser;
use crate::upstream::{Answer, answer, caller};

#[get("")]
async fn get_usage(user: AuthUser, req: HttpRequest, engine: web::Data<Proxy>) -> Answer {
    let path = match req.query_string() {
        "" => "/usage".to_string(),
        query => format!("/usage?{query}"),
    };
    answer(engine.get(&caller(&user), &path).await?)
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(get_usage);
}

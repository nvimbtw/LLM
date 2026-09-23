//! Built vocabularies. Listing one is reading; deleting one takes the artifacts
//! a run may still be training against with it, so that stays with an admin.

use actix_web::{delete, get, web, HttpRequest};
use llm_engine::Proxy;
use uuid::Uuid;

use crate::extractors::{AdminUser, AuthUser};
use crate::upstream::{answer, caller, with_query, Answer};

#[get("")]
async fn list(user: AuthUser, req: HttpRequest, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .get(&caller(&user), &with_query("/tokenizers", &req))
            .await?,
    )
}

#[get("/{tokenizer_id}")]
async fn get_one(user: AuthUser, id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .get(&caller(&user), &format!("/tokenizers/{id}"))
            .await?,
    )
}

#[delete("/{tokenizer_id}")]
async fn remove(_admin: AdminUser, id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .delete(&auth::Caller::System, &format!("/tokenizers/{id}"))
            .await?,
    )
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(list);
    cfg.service(get_one);
    cfg.service(remove);
}

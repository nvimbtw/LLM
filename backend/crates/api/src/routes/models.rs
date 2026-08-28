//! Trained models. A model is shared, so changing one is admin work.

use actix_web::{delete, get, patch, post, web};
use engine::Proxy;
use serde_json::Value;
use uuid::Uuid;

use crate::extractors::{AdminUser, AuthUser};
use crate::upstream::{Answer, answer, caller};

#[get("")]
async fn list(user: AuthUser, engine: web::Data<Proxy>) -> Answer {
    answer(engine.get(&caller(&user), "/models").await?)
}

#[get("/{model_id}")]
async fn get_one(user: AuthUser, model_id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(engine.get(&caller(&user), &format!("/models/{model_id}")).await?)
}

#[get("/{model_id}/benchmarks")]
async fn benchmarks(
    user: AuthUser,
    model_id: web::Path<Uuid>,
    engine: web::Data<Proxy>,
) -> Answer {
    answer(engine.get(&caller(&user), &format!("/models/{model_id}/benchmarks")).await?)
}

/// One head's attention over the caller's own text, materialized by the engine
/// for this request and nothing else.
///
/// `AuthUser` rather than `AdminUser`, on the same reasoning `/tokenize/count`
/// is: this is one forward pass over a model any signed-in person can already
/// hold a conversation with, and reading where its heads look is strictly less
/// than that. What keeps a quadratic response off an open route is the engine's
/// own bound on the sequence, which answers 400 rather than a large matrix.
#[post("/{model_id}/attention")]
async fn attention(
    user: AuthUser,
    model_id: web::Path<Uuid>,
    body: web::Json<Value>,
    engine: web::Data<Proxy>,
) -> Answer {
    let path = format!("/models/{model_id}/attention");
    answer(engine.post(&caller(&user), &path, body.into_inner()).await?)
}

#[patch("/{model_id}")]
async fn edit(
    _admin: AdminUser,
    model_id: web::Path<Uuid>,
    body: web::Json<Value>,
    engine: web::Data<Proxy>,
) -> Answer {
    let path = format!("/models/{model_id}");
    answer(engine.patch(&auth::Caller::System, &path, body.into_inner()).await?)
}

#[delete("/{model_id}")]
async fn remove(_admin: AdminUser, model_id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(engine.delete(&auth::Caller::System, &format!("/models/{model_id}")).await?)
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(list);
    cfg.service(benchmarks);
    cfg.service(get_one);
    cfg.service(attention);
    cfg.service(edit);
    cfg.service(remove);
}

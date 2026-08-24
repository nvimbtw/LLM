//! Trained models. A model is shared, so changing one is admin work.

use actix_web::{delete, get, patch, web};
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
    bench: web::Data<bench_engine::Proxy>,
    engine: web::Data<Proxy>,
) -> Answer {
    let caller = caller(&user);
    let path = format!("/models/{model_id}/benchmarks");
    // New benchmarks live in bench-engine; fall back to engine for old rows
    match bench.get(&caller, &path).await {
        Ok(v) => answer(v),
        Err(service::Error::Status { code: 404, .. }) => answer(engine.get(&caller, &path).await?),
        Err(e) if matches!(e, service::Error::Transport(_) | service::Error::Decode(_)) => {
            // Bench-engine down — try engine's old data
            match engine.get(&caller, &path).await {
                Ok(v) => answer(v),
                Err(_) => Err(e.into()),
            }
        }
        Err(e) => Err(e.into()),
    }
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
    cfg.service(edit);
    cfg.service(remove);
}

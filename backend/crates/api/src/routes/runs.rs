//! The queue. One device, one line — and the engine is the one holding it.

use actix_web::{HttpRequest, delete, get, patch, post, web};
use engine::Proxy;
use serde_json::Value;
use uuid::Uuid;

use crate::extractors::{AdminUser, AuthUser};
use crate::upstream::{Answer, answer, caller, with_query};

#[get("")]
async fn list(user: AuthUser, req: HttpRequest, engine: web::Data<Proxy>) -> Answer {
    answer(engine.get(&caller(&user), &with_query("/runs", &req)).await?)
}

#[get("/{run_id}")]
async fn get_one(user: AuthUser, run_id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(engine.get(&caller(&user), &format!("/runs/{run_id}")).await?)
}

#[post("/train")]
async fn train(admin: AdminUser, body: web::Json<Value>, engine: web::Data<Proxy>) -> Answer {
    answer(engine.post(&started_by(&admin), "/train", body.into_inner()).await?)
}

#[post("/sft")]
async fn sft(admin: AdminUser, body: web::Json<Value>, engine: web::Data<Proxy>) -> Answer {
    answer(engine.post(&started_by(&admin), "/sft", body.into_inner()).await?)
}

/// A sequence, each step running against the model the one before it produced.
#[post("/chain")]
async fn chain(admin: AdminUser, body: web::Json<Value>, engine: web::Data<Proxy>) -> Answer {
    answer(engine.post(&started_by(&admin), "/runs/chain", body.into_inner()).await?)
}

#[post("/benchmark")]
async fn benchmark(admin: AdminUser, body: web::Json<Value>, engine: web::Data<Proxy>) -> Answer {
    answer(engine.post(&started_by(&admin), "/eval", body.into_inner()).await?)
}

#[post("/{run_id}/cancel")]
async fn cancel(_admin: AdminUser, run_id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    let path = format!("/runs/{run_id}/cancel");
    answer(engine.post(&auth::Caller::System, &path, Value::Null).await?)
}

#[patch("/{run_id}")]
async fn reorder(
    _admin: AdminUser,
    run_id: web::Path<Uuid>,
    req: HttpRequest,
    engine: web::Data<Proxy>,
) -> Answer {
    let path = with_query(&format!("/runs/{run_id}"), &req);
    answer(engine.patch(&auth::Caller::System, &path, Value::Null).await?)
}

#[post("/pause")]
async fn pause(_admin: AdminUser, req: HttpRequest, engine: web::Data<Proxy>) -> Answer {
    let path = with_query("/runs/pause", &req);
    answer(engine.post(&auth::Caller::System, &path, Value::Null).await?)
}

#[delete("/{run_id}")]
async fn remove(_admin: AdminUser, run_id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(engine.delete(&auth::Caller::System, &format!("/runs/{run_id}")).await?)
}

fn started_by(admin: &AdminUser) -> auth::Caller {
    auth::Caller::User(admin.claims.sub)
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    // The literal segments come first, so "/pause" is not read as a run id.
    cfg.service(train);
    cfg.service(sft);
    cfg.service(benchmark);
    cfg.service(chain);
    cfg.service(pause);
    cfg.service(list);
    cfg.service(get_one);
    cfg.service(cancel);
    cfg.service(reorder);
    cfg.service(remove);
}

//! Scores across models — the history page's question, which names no model.

use actix_web::{delete, get, web, HttpRequest};
use bench_engine::Proxy;
use uuid::Uuid;

use crate::extractors::{AdminUser, AuthUser};
use crate::upstream::{answer, caller, with_query, Answer};

#[get("")]
async fn list(user: AuthUser, req: HttpRequest, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .get(&caller(&user), &with_query("/benchmarks", &req))
            .await?,
    )
}

#[get("/suite")]
async fn suite(user: AuthUser, engine: web::Data<Proxy>) -> Answer {
    answer(engine.get(&caller(&user), "/benchmarks/suite").await?)
}

#[get("/{benchmark_id}")]
async fn get_one(user: AuthUser, id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .get(&caller(&user), &format!("/benchmarks/{id}"))
            .await?,
    )
}

#[delete("/{benchmark_id}")]
async fn remove(_admin: AdminUser, id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .delete(&auth::Caller::System, &format!("/benchmarks/{id}"))
            .await?,
    )
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    // Before "/{benchmark_id}", so the literal segment is not read as an id.
    cfg.service(suite);
    cfg.service(list);
    cfg.service(get_one);
    cfg.service(remove);
}

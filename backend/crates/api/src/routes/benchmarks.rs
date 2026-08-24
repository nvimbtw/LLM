use actix_web::{HttpRequest, delete, get, web};
use uuid::Uuid;

use crate::extractors::{AdminUser, AuthUser};
use crate::upstream::{Answer, answer, caller, with_query};

#[get("")]
async fn list(
    user: AuthUser,
    req: HttpRequest,
    bench: web::Data<bench_engine::Proxy>,
    engine: web::Data<engine::Proxy>,
) -> Answer {
    // language lives in bench-engine, markup/assistant still in engine for now
    let query = req.query_string();
    let is_language = query.contains("suite=language") || !query.contains("suite=");
    if is_language {
        let bench_res = bench.get(&caller(&user), &with_query("/benchmarks", &req)).await;
        let engine_res = engine.get(&caller(&user), &with_query("/benchmarks", &req)).await;
        match (bench_res, engine_res) {
            (Ok(b), Ok(e)) => {
                // merge both lists, bench first (newer)
                let merged = merge_benchmark_lists(b, e);
                answer(merged)
            }
            (Ok(b), Err(_)) => answer(b),
            (Err(_), Ok(e)) => answer(e),
            (Err(e), Err(_)) => Err(e.into()),
        }
    } else {
        answer(engine.get(&caller(&user), &with_query("/benchmarks", &req)).await?)
    }
}

#[get("/suite")]
async fn suite(user: AuthUser, bench: web::Data<bench_engine::Proxy>) -> Answer {
    answer(bench.get(&caller(&user), "/benchmarks/suite").await?)
}

#[get("/{benchmark_id}")]
async fn get_one(
    user: AuthUser,
    id: web::Path<Uuid>,
    bench: web::Data<bench_engine::Proxy>,
    engine: web::Data<engine::Proxy>,
) -> Answer {
    let caller = caller(&user);
    let path = format!("/benchmarks/{id}");
    match bench.get(&caller, &path).await {
        Ok(v) => answer(v),
        Err(service::Error::Status { code: 404, .. }) => answer(engine.get(&caller, &path).await?),
        Err(e) => Err(e.into()),
    }
}

#[delete("/{benchmark_id}")]
async fn remove(
    _admin: AdminUser,
    id: web::Path<Uuid>,
    bench: web::Data<bench_engine::Proxy>,
    engine: web::Data<engine::Proxy>,
) -> Answer {
    let sys = auth::Caller::System;
    let path = format!("/benchmarks/{id}");
    match bench.delete(&sys, &path).await {
        Ok(v) => answer(v),
        Err(service::Error::Status { code: 404, .. }) => answer(engine.delete(&sys, &path).await?),
        Err(e) => Err(e.into()),
    }
}

fn merge_benchmark_lists(bench: serde_json::Value, engine: serde_json::Value) -> serde_json::Value {
    let b = bench.get("benchmarks").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let e = engine.get("benchmarks").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let mut merged = b;
    merged.extend(e);
    // sort by created_at desc
    merged.sort_by(|a, b| {
        let ca = a.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
        let cb = b.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
        cb.cmp(ca)
    });
    serde_json::json!({ "benchmarks": merged })
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(suite);
    cfg.service(list);
    cfg.service(get_one);
    cfg.service(remove);
}

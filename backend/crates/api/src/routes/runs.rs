use actix_web::{HttpRequest, delete, get, patch, post, web};
use serde_json::Value;
use uuid::Uuid;

use crate::extractors::{AdminUser, AuthUser};
use crate::upstream::{Answer, answer, caller, with_query};

#[get("")]
async fn list(
    user: AuthUser,
    req: HttpRequest,
    engine: web::Data<engine::Proxy>,
    bench: web::Data<bench_engine::Proxy>,
) -> Answer {
    let query = with_query("/runs", &req);
    let caller = caller(&user);
    // Try to merge both boards. If bench is down, engine alone is still useful.
    let engine_res = engine.get(&caller, &query).await;
    let bench_res = bench.get(&caller, &query).await;
    match (engine_res, bench_res) {
        (Ok(eng), Ok(bch)) => {
            let merged = merge_boards(eng, bch);
            answer(merged)
        }
        (Ok(eng), Err(_)) => answer(eng),
        (Err(_), Ok(bch)) => answer(bch),
        (Err(e), Err(_)) => Err(e.into()),
    }
}

#[get("/{run_id}")]
async fn get_one(
    user: AuthUser,
    run_id: web::Path<Uuid>,
    engine: web::Data<engine::Proxy>,
    bench: web::Data<bench_engine::Proxy>,
) -> Answer {
    let caller = caller(&user);
    let path = format!("/runs/{run_id}");
    match engine.get(&caller, &path).await {
        Ok(v) => answer(v),
        Err(service::Error::Status { code: 404, .. }) => {
            answer(bench.get(&caller, &path).await?)
        }
        Err(e) => Err(e.into()),
    }
}

#[post("/train")]
async fn train(admin: AdminUser, body: web::Json<Value>, engine: web::Data<engine::Proxy>) -> Answer {
    answer(engine.post(&started_by(&admin), "/train", body.into_inner()).await?)
}

#[post("/sft")]
async fn sft(admin: AdminUser, body: web::Json<Value>, engine: web::Data<engine::Proxy>) -> Answer {
    answer(engine.post(&started_by(&admin), "/sft", body.into_inner()).await?)
}

#[post("/tokenize")]
async fn tokenize(admin: AdminUser, body: web::Json<Value>, engine: web::Data<engine::Proxy>) -> Answer {
    answer(engine.post(&started_by(&admin), "/tokenize", body.into_inner()).await?)
}

#[post("/chain")]
async fn chain(admin: AdminUser, body: web::Json<Value>, engine: web::Data<engine::Proxy>) -> Answer {
    answer(engine.post(&started_by(&admin), "/runs/chain", body.into_inner()).await?)
}

#[post("/benchmark")]
async fn benchmark(admin: AdminUser, body: web::Json<Value>, bench: web::Data<bench_engine::Proxy>) -> Answer {
    answer(bench.post(&started_by(&admin), "/eval", body.into_inner()).await?)
}

#[post("/{run_id}/cancel")]
async fn cancel(
    _admin: AdminUser,
    run_id: web::Path<Uuid>,
    engine: web::Data<engine::Proxy>,
    bench: web::Data<bench_engine::Proxy>,
) -> Answer {
    let path = format!("/runs/{run_id}/cancel");
    let sys = auth::Caller::System;
    match engine.post(&sys, &path, Value::Null).await {
        Ok(v) => answer(v),
        Err(service::Error::Status { code: 404, .. }) => answer(bench.post(&sys, &path, Value::Null).await?),
        Err(e) => Err(e.into()),
    }
}

#[patch("/{run_id}")]
async fn reorder(
    _admin: AdminUser,
    run_id: web::Path<Uuid>,
    req: HttpRequest,
    engine: web::Data<engine::Proxy>,
    bench: web::Data<bench_engine::Proxy>,
) -> Answer {
    let sys = auth::Caller::System;
    let path_e = with_query(&format!("/runs/{run_id}"), &req);
    let path_b = path_e.clone();
    match engine.patch(&sys, &path_e, Value::Null).await {
        Ok(v) => answer(v),
        Err(service::Error::Status { code: 404, .. }) => answer(bench.patch(&sys, &path_b, Value::Null).await?),
        Err(e) => Err(e.into()),
    }
}

#[post("/pause")]
async fn pause(
    _admin: AdminUser,
    req: HttpRequest,
    engine: web::Data<engine::Proxy>,
    bench: web::Data<bench_engine::Proxy>,
) -> Answer {
    let path = with_query("/runs/pause", &req);
    let sys = auth::Caller::System;
    // Pause both queues; frontend only shows one paused flag.
    let e_res = engine.post(&sys, &path, Value::Null).await;
    let b_res = bench.post(&sys, &path, Value::Null).await;
    match (e_res, b_res) {
        (Ok(v), _) => answer(v),
        (Err(_), Ok(v)) => answer(v),
        (Err(e), Err(_)) => Err(e.into()),
    }
}

#[delete("/{run_id}")]
async fn remove(
    _admin: AdminUser,
    run_id: web::Path<Uuid>,
    engine: web::Data<engine::Proxy>,
    bench: web::Data<bench_engine::Proxy>,
) -> Answer {
    let path = format!("/runs/{run_id}");
    let sys = auth::Caller::System;
    match engine.delete(&sys, &path).await {
        Ok(v) => answer(v),
        Err(service::Error::Status { code: 404, .. }) => answer(bench.delete(&sys, &path).await?),
        Err(e) => Err(e.into()),
    }
}

fn merge_boards(engine: Value, bench: Value) -> Value {
    let mut eng = engine.clone();
    let bch = bench;
    // Merge active: concatenate
    if let (Some(ea), Some(ba)) = (eng.get("active").and_then(|v| v.as_array()).cloned(), bch.get("active").and_then(|v| v.as_array()).cloned()) {
        let mut merged = ea;
        merged.extend(ba);
        if let Some(obj) = eng.as_object_mut() {
            obj.insert("active".to_string(), Value::Array(merged));
        }
    }
    // Merge history: concatenate and sort by created_at desc
    let b_history_total = bch.get("history_total").and_then(|v| v.as_u64()).unwrap_or(0);
    if let (Some(eh), Some(bh)) = (eng.get("history").and_then(|v| v.as_array()).cloned(), bch.get("history").and_then(|v| v.as_array()).cloned()) {
        let mut merged = eh;
        merged.extend(bh);
        merged.sort_by(|a, b| {
            let ca = a.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
            let cb = b.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
            cb.cmp(ca)
        });
        let e_history_total = eng.get("history_total").and_then(|v| v.as_u64()).unwrap_or(0);
        if let Some(obj) = eng.as_object_mut() {
            obj.insert("history".to_string(), Value::Array(merged));
            obj.insert("history_total".to_string(), Value::Number((e_history_total + b_history_total).into()));
        }
    }
    // paused if either
    let ep = eng.get("paused").and_then(|v| v.as_bool()).unwrap_or(false);
    let bp = bch.get("paused").and_then(|v| v.as_bool()).unwrap_or(false);
    if let Some(obj) = eng.as_object_mut() {
        obj.insert("paused".to_string(), Value::Bool(ep || bp));
    }
    eng
}

fn started_by(admin: &AdminUser) -> auth::Caller {
    auth::Caller::User(admin.claims.sub)
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(train);
    cfg.service(sft);
    cfg.service(tokenize);
    cfg.service(benchmark);
    cfg.service(chain);
    cfg.service(pause);
    cfg.service(list);
    cfg.service(get_one);
    cfg.service(cancel);
    cfg.service(reorder);
    cfg.service(remove);
}

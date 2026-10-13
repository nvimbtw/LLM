//! The queue. One device, one line — and the LLM engine is the one holding it.

use actix_web::{delete, get, patch, post, web, HttpRequest};
use llm_engine::Proxy;
use serde_json::Value;
use uuid::Uuid;

use crate::extractors::{AdminUser, AuthUser};
use crate::upstream::{answer, caller, with_query, Answer};

#[get("")]
async fn list(user: AuthUser, req: HttpRequest, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .get(&caller(&user), &with_query("/runs", &req))
            .await?,
    )
}

#[get("/{run_id}")]
async fn get_one(user: AuthUser, run_id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .get(&caller(&user), &format!("/runs/{run_id}"))
            .await?,
    )
}

#[post("/train")]
async fn train(admin: AdminUser, body: web::Json<Value>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .post(&started_by(&admin), "/train", body.into_inner())
            .await?,
    )
}

#[post("/sft")]
async fn sft(admin: AdminUser, body: web::Json<Value>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .post(&started_by(&admin), "/sft", body.into_inner())
            .await?,
    )
}

/// Practising against a gym: the run rolls episodes out, grades them, and takes
/// a policy-gradient step on the tokens the model itself drew.
///
/// A route of its own rather than a flag on `/train` for the reason the LLM engine
/// gives it a run kind of its own — what it reports is a pass rate where a
/// training run reports a loss, and nothing reading a run's frames could tell
/// those apart from the numbers.
#[post("/rl")]
async fn rl(admin: AdminUser, body: web::Json<Value>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .post(&started_by(&admin), "/rl", body.into_inner())
            .await?,
    )
}

/// A vocabulary, built once and named by the runs that train against it.
#[post("/tokenize")]
async fn tokenize(admin: AdminUser, body: web::Json<Value>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .post(&started_by(&admin), "/tokenize", body.into_inner())
            .await?,
    )
}

/// A sequence, each step running against the model the one before it produced.
#[post("/chain")]
async fn chain(admin: AdminUser, body: web::Json<Value>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .post(&started_by(&admin), "/runs/chain", body.into_inner())
            .await?,
    )
}

#[post("/benchmark")]
async fn benchmark(admin: AdminUser, body: web::Json<Value>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .post(&started_by(&admin), "/eval", body.into_inner())
            .await?,
    )
}

#[post("/{run_id}/cancel")]
async fn cancel(_admin: AdminUser, run_id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    let path = format!("/runs/{run_id}/cancel");
    answer(
        engine
            .post(&auth::Caller::System, &path, Value::Null)
            .await?,
    )
}

#[post("/{run_id}/recover")]
async fn recover(admin: AdminUser, run_id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .post(
                &started_by(&admin),
                &format!("/runs/{run_id}/recover"),
                Value::Null,
            )
            .await?,
    )
}

#[post("/{run_id}/retry")]
async fn retry(admin: AdminUser, run_id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .post(
                &started_by(&admin),
                &format!("/runs/{run_id}/retry"),
                Value::Null,
            )
            .await?,
    )
}

#[patch("/{run_id}")]
async fn reorder(
    _admin: AdminUser,
    run_id: web::Path<Uuid>,
    req: HttpRequest,
    engine: web::Data<Proxy>,
) -> Answer {
    let path = with_query(&format!("/runs/{run_id}"), &req);
    answer(
        engine
            .patch(&auth::Caller::System, &path, Value::Null)
            .await?,
    )
}

#[post("/pause")]
async fn pause(_admin: AdminUser, req: HttpRequest, engine: web::Data<Proxy>) -> Answer {
    let path = with_query("/runs/pause", &req);
    answer(
        engine
            .post(&auth::Caller::System, &path, Value::Null)
            .await?,
    )
}

#[delete("/{run_id}")]
async fn remove(_admin: AdminUser, run_id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .delete(&auth::Caller::System, &format!("/runs/{run_id}"))
            .await?,
    )
}

fn started_by(admin: &AdminUser) -> auth::Caller {
    auth::Caller::User(admin.claims.sub)
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    // The literal segments come first, so "/pause" is not read as a run id.
    cfg.service(train);
    cfg.service(sft);
    cfg.service(rl);
    cfg.service(tokenize);
    cfg.service(benchmark);
    cfg.service(chain);
    cfg.service(pause);
    cfg.service(list);
    cfg.service(get_one);
    cfg.service(cancel);
    cfg.service(recover);
    cfg.service(retry);
    cfg.service(reorder);
    cfg.service(remove);
}

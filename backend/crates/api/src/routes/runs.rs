//! The queue. One device, one line — and the LLM engine is the one holding it.

use actix_web::{delete, get, patch, post, web, HttpRequest};
use llm_engine::{Method, Proxy};
use serde_json::Value;
use uuid::Uuid;

use crate::extractors::{AdminUser, AuthUser};
use crate::upstream::{admin_json, answer, visible_get, with_query, Answer};

/// Read visible active jobs and filtered history totals using verified authorization.
#[get("")]
async fn list(user: AuthUser, req: HttpRequest, engine: web::Data<Proxy>) -> Answer {
    visible_get(&user, &engine, &with_query("/runs", &req)).await
}

/// Read one job only after the engine applies the signed visibility policy.
#[get("/{run_id}")]
async fn get_one(user: AuthUser, run_id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    visible_get(&user, &engine, &format!("/runs/{run_id}")).await
}

/// Forward opaque pretraining configuration, retry identity and verified administrative scope.
#[post("/train")]
async fn train(
    admin: AdminUser,
    req: HttpRequest,
    body: web::Json<Value>,
    engine: web::Data<Proxy>,
) -> Answer {
    submit(&admin, &req, &engine, "/train", body.into_inner()).await
}

/// Forward inherited SFT configuration with retry identity and verified administrative scope.
#[post("/sft")]
async fn sft(
    admin: AdminUser,
    req: HttpRequest,
    body: web::Json<Value>,
    engine: web::Data<Proxy>,
) -> Answer {
    submit(&admin, &req, &engine, "/sft", body.into_inner()).await
}

/// Forward inherited RL configuration with retry identity and verified administrative scope.
#[post("/rl")]
async fn rl(
    admin: AdminUser,
    req: HttpRequest,
    body: web::Json<Value>,
    engine: web::Data<Proxy>,
) -> Answer {
    submit(&admin, &req, &engine, "/rl", body.into_inner()).await
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
async fn chain(
    admin: AdminUser,
    req: HttpRequest,
    body: web::Json<Value>,
    engine: web::Data<Proxy>,
) -> Answer {
    submit(&admin, &req, &engine, "/runs/chain", body.into_inner()).await
}

#[post("/benchmark")]
async fn benchmark(admin: AdminUser, body: web::Json<Value>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .post(&started_by(&admin), "/eval", body.into_inner())
            .await?,
    )
}

/// Cancel a private or ordinary job with verified administrative visibility and the originating actor.
#[post("/{run_id}/cancel")]
async fn cancel(
    admin: AdminUser,
    run_id: web::Path<Uuid>,
    req: HttpRequest,
    engine: web::Data<Proxy>,
) -> Answer {
    submit(
        &admin,
        &req,
        &engine,
        &format!("/runs/{run_id}/cancel"),
        Value::Null,
    )
    .await
}

/// Forward exact recovery under the verified actor, preserving the engine's retry identity and status.
#[post("/{run_id}/recover")]
async fn recover(
    admin: AdminUser,
    run_id: web::Path<Uuid>,
    req: HttpRequest,
    engine: web::Data<Proxy>,
) -> Answer {
    submit(
        &admin,
        &req,
        &engine,
        &format!("/runs/{run_id}/recover"),
        Value::Null,
    )
    .await
}

/// Forward a new attempt over frozen job inputs without interpreting engine-owned recipe data.
#[post("/{run_id}/retry")]
async fn retry(
    admin: AdminUser,
    run_id: web::Path<Uuid>,
    req: HttpRequest,
    engine: web::Data<Proxy>,
) -> Answer {
    submit(
        &admin,
        &req,
        &engine,
        &format!("/runs/{run_id}/retry"),
        Value::Null,
    )
    .await
}

/// Reorder a queued job without dropping the administrative scope needed to see validation rows.
#[patch("/{run_id}")]
async fn reorder(
    admin: AdminUser,
    run_id: web::Path<Uuid>,
    req: HttpRequest,
    engine: web::Data<Proxy>,
) -> Answer {
    let path = with_query(&format!("/runs/{run_id}"), &req);
    admin_json(&admin, &engine, Method::PATCH, &path, None).await
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

/// Delete a finished job with administrative visibility, leaving lifecycle/refusal decisions in the engine.
#[delete("/{run_id}")]
async fn remove(admin: AdminUser, run_id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    admin_json(
        &admin,
        &engine,
        Method::DELETE,
        &format!("/runs/{run_id}"),
        None,
    )
    .await
}

/// Preserve bounded engine status/body while signing actor and scope and forwarding query values unchanged.
async fn submit(
    admin: &AdminUser,
    req: &HttpRequest,
    engine: &Proxy,
    path: &str,
    body: Value,
) -> Answer {
    admin_json(
        admin,
        engine,
        Method::POST,
        &with_query(path, req),
        Some(body),
    )
    .await
}

/// Attribute legacy queue operations to the verified administrator's user UUID.
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

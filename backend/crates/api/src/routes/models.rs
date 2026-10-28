//! Trained models. A model is shared, so changing one is admin work.

use actix_web::{delete, get, patch, post, web, HttpRequest};
use llm_engine::Proxy;
use serde_json::Value;
use uuid::Uuid;

use crate::extractors::{AdminUser, AuthUser, ModelReader};
use crate::upstream::{answer, caller, visible_get, with_query, Answer};

/// The query string rides through so `?checkpoints=true` reaches the LLM engine,
/// which leaves a run's intermediates out unless asked. Nothing here restates
/// that default: the LLM engine owns it, and a second copy could disagree with it.
#[get("")]
async fn list(user: ModelReader, req: HttpRequest, engine: web::Data<Proxy>) -> Answer {
    model_read(&user, &engine, &with_query("/models", &req)).await
}

/// Read authorized model facts; attach existing benchmark records only for administrators.
#[get("/{model_id}")]
async fn get_one(
    user: ModelReader,
    model_id: web::Path<Uuid>,
    engine: web::Data<Proxy>,
    bench: web::Data<bench_engine::Proxy>,
) -> Answer {
    let (status, mut model) = engine
        .scoped_json(
            &user.caller(),
            user.visibility(),
            llm_engine::Method::GET,
            &format!("/models/{model_id}"),
            None,
            8 * 1024 * 1024,
        )
        .await?;
    if !(200..300).contains(&status) {
        return Ok(actix_web::HttpResponse::build(
            actix_web::http::StatusCode::from_u16(status)
                .unwrap_or(actix_web::http::StatusCode::BAD_GATEWAY),
        )
        .json(model));
    }
    if user.visibility() != "administration" {
        model["benchmarks"] = serde_json::json!([]);
        return answer(model);
    }
    let scores = bench
        .get(
            &user.caller(),
            &format!("/benchmarks?model_id={model_id}&limit=50"),
        )
        .await?;
    model["benchmarks"] = scores["benchmarks"].clone();
    answer(model)
}

/// Read benchmark records only after the engine authorizes the selected model.
#[get("/{model_id}/benchmarks")]
async fn benchmarks(
    user: AuthUser,
    model_id: web::Path<Uuid>,
    engine: web::Data<Proxy>,
    bench: web::Data<bench_engine::Proxy>,
) -> Answer {
    let admitted = visible_get(&user, &engine, &format!("/models/{model_id}")).await?;
    if !admitted.status().is_success() {
        return Ok(admitted);
    }
    answer(
        bench
            .get(
                &caller(&user),
                &format!("/benchmarks?model_id={model_id}&limit=50"),
            )
            .await?,
    )
}

/// Return visible published checkpoints, or all run intermediates for an administrator.
#[get("/{model_id}/checkpoints")]
async fn checkpoints(
    user: ModelReader,
    model_id: web::Path<Uuid>,
    engine: web::Data<Proxy>,
) -> Answer {
    model_read(&user, &engine, &format!("/models/{model_id}/checkpoints")).await
}

/// One head's attention over the caller's own text, materialized by the LLM engine
/// for this request and nothing else.
///
/// `AuthUser` rather than `AdminUser`, on the same reasoning `/tokenize/count`
/// is: this is one forward pass over a model any signed-in person can already
/// hold a conversation with, and reading where its heads look is strictly less
/// than that. What keeps a quadratic response off an open route is the LLM engine's
/// own bound on the sequence, which answers 400 rather than a large matrix.
#[post("/{model_id}/attention")]
async fn attention(
    user: AuthUser,
    model_id: web::Path<Uuid>,
    body: web::Json<Value>,
    engine: web::Data<Proxy>,
) -> Answer {
    let admitted = visible_get(&user, &engine, &format!("/models/{model_id}")).await?;
    if !admitted.status().is_success() {
        return Ok(admitted);
    }
    let path = format!("/models/{model_id}/attention");
    answer(
        engine
            .post(&caller(&user), &path, body.into_inner())
            .await?,
    )
}

/// Discover actual legacy dimensions or declared occurrence-aware attention views for any signed-in user.
#[get("/{model_id}/attention-points")]
async fn attention_points(
    user: AuthUser,
    model_id: web::Path<Uuid>,
    engine: web::Data<Proxy>,
) -> Answer {
    let admitted = visible_get(&user, &engine, &format!("/models/{model_id}")).await?;
    if !admitted.status().is_success() {
        return Ok(admitted);
    }
    visible_get(
        &user,
        &engine,
        &format!("/models/{model_id}/attention-points"),
    )
    .await
}

/// Persist model display/publication fields after administrator admission.
#[patch("/{model_id}")]
async fn edit(
    _admin: AdminUser,
    model_id: web::Path<Uuid>,
    body: web::Json<Value>,
    engine: web::Data<Proxy>,
) -> Answer {
    let path = format!("/models/{model_id}");
    answer(
        engine
            .patch(&auth::Caller::System, &path, body.into_inner())
            .await?,
    )
}

/// Remove model artifacts after administrator admission.
#[delete("/{model_id}")]
async fn remove(_admin: AdminUser, model_id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .delete(&auth::Caller::System, &format!("/models/{model_id}"))
            .await?,
    )
}

/// Forward model reads with verified optional identity, trusted visibility and the exact upstream status.
async fn model_read(reader: &ModelReader, engine: &Proxy, path: &str) -> Answer {
    let (status, value) = engine
        .scoped_json(
            &reader.caller(),
            reader.visibility(),
            llm_engine::Method::GET,
            path,
            None,
            8 * 1024 * 1024,
        )
        .await?;
    Ok(actix_web::HttpResponse::build(
        actix_web::http::StatusCode::from_u16(status)
            .unwrap_or(actix_web::http::StatusCode::BAD_GATEWAY),
    )
    .json(value))
}

/// Register shared model reads/inspection and administrator-only model mutations.
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(list);
    cfg.service(benchmarks);
    cfg.service(checkpoints);
    cfg.service(get_one);
    cfg.service(attention);
    cfg.service(attention_points);
    cfg.service(edit);
    cfg.service(remove);
}

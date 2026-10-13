//! Trained models. A model is shared, so changing one is admin work.

use actix_web::{delete, get, patch, post, web, HttpRequest};
use llm_engine::Proxy;
use serde_json::Value;
use uuid::Uuid;

use crate::extractors::{AdminUser, AuthUser};
use crate::upstream::{answer, caller, with_query, Answer};

/// The query string rides through so `?checkpoints=true` reaches the LLM engine,
/// which leaves a run's intermediates out unless asked. Nothing here restates
/// that default: the LLM engine owns it, and a second copy could disagree with it.
#[get("")]
async fn list(user: AuthUser, req: HttpRequest, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .get(&caller(&user), &with_query("/models", &req))
            .await?,
    )
}

#[get("/{model_id}")]
async fn get_one(
    user: AuthUser,
    model_id: web::Path<Uuid>,
    engine: web::Data<Proxy>,
    bench: web::Data<bench_engine::Proxy>,
) -> Answer {
    let mut model = engine
        .get(&caller(&user), &format!("/models/{model_id}"))
        .await?;
    let scores = bench
        .get(
            &caller(&user),
            &format!("/benchmarks?model_id={model_id}&limit=50"),
        )
        .await?;
    model["benchmarks"] = scores["benchmarks"].clone();
    answer(model)
}

#[get("/{model_id}/benchmarks")]
async fn benchmarks(
    user: AuthUser,
    model_id: web::Path<Uuid>,
    bench: web::Data<bench_engine::Proxy>,
) -> Answer {
    answer(
        bench
            .get(
                &caller(&user),
                &format!("/benchmarks?model_id={model_id}&limit=50"),
            )
            .await?,
    )
}

/// A model's intermediate checkpoints, in the order the run took them.
///
/// `AuthUser`, like the model detail and the attention pass next door and unlike
/// the edit and delete below: a checkpoint is a model, and reading which ones a
/// run left behind reveals nothing an ordinary session cannot already read off
/// the model itself. Admin is for the two routes that change what everyone else
/// sees.
#[get("/{model_id}/checkpoints")]
async fn checkpoints(
    user: AuthUser,
    model_id: web::Path<Uuid>,
    engine: web::Data<Proxy>,
) -> Answer {
    answer(
        engine
            .get(&caller(&user), &format!("/models/{model_id}/checkpoints"))
            .await?,
    )
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
    let path = format!("/models/{model_id}/attention");
    answer(
        engine
            .post(&caller(&user), &path, body.into_inner())
            .await?,
    )
}

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

#[delete("/{model_id}")]
async fn remove(_admin: AdminUser, model_id: web::Path<Uuid>, engine: web::Data<Proxy>) -> Answer {
    answer(
        engine
            .delete(&auth::Caller::System, &format!("/models/{model_id}"))
            .await?,
    )
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(list);
    cfg.service(benchmarks);
    cfg.service(checkpoints);
    cfg.service(get_one);
    cfg.service(attention);
    cfg.service(edit);
    cfg.service(remove);
}

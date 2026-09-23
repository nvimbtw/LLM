//! What a service may ask the backend for.
//!
//! One direction, one consumer: the LLM engine. It reads the documents a corpus
//! is packed from, and it drives the gym episodes a reinforcement-learning batch
//! is rolled out over. Both belong to another Python service, and the two Python
//! services never speak — so the request arrives here and is re-minted, carrying
//! the same acting caller onward.
//!
//! Nothing here is reachable with a browser session. `ServiceCaller` accepts a
//! service token and only a service token, which is why these routes live under
//! their own scope rather than beside the `/data` and `/gym` routes a person
//! uses: the same path with two authentication schemes is the kind of thing
//! that gets one of them wrong.
//!
//! **Datasets are reads only.** A run consumes them; it does not create, clean
//! or delete them, and a service that cannot write cannot corrupt what a person
//! curated. The gym family is not read-only and cannot be — a rollout runs
//! commands and grades answers — but every one of its writes lands in the gyms
//! service's own record, which is the module that owns them.
//!
//! **The LLM engine never sees a `{stdout, stderr, exit_code}` triple.** The step
//! route answers with the *rendered* observation, from `gym::play`, which is the
//! same renderer the browser's socket loop and the live chat `bash` tool use.
//! That is the whole reason an engine-driven rollout is allowed to exist at all:
//! it makes the round trip the sandbox already forced it into pay for the
//! rendering too. See
//! `docs/decisions/reinforcement-learning-in-the-llm-engine.md`.

use actix_web::{get, post, web, HttpRequest};
use data_engine::Proxy;
use serde_json::Value;

use crate::extractors::ServiceCaller;
use crate::gym;
use crate::upstream::{answer, with_query, Answer};

#[get("/datasets/{id}")]
async fn dataset(service: ServiceCaller, id: web::Path<String>, data: web::Data<Proxy>) -> Answer {
    answer(
        data.get(&service.caller, &format!("/datasets/{id}"))
            .await?,
    )
}

#[get("/gyms")]
async fn catalogue(service: ServiceCaller, gyms: web::Data<gyms_engine::Proxy>) -> Answer {
    answer(gyms.get(&service.caller, "/gyms").await?)
}

/// One page of documents. The query string is passed through untouched: paging
/// is the data engine's to define, and a run sizes its own pages by what the
/// last one measured.
#[get("/datasets/{id}/documents")]
async fn documents(
    service: ServiceCaller,
    id: web::Path<String>,
    req: HttpRequest,
    data: web::Data<Proxy>,
) -> Answer {
    let path = with_query(&format!("/datasets/{id}/documents"), &req);
    answer(data.get(&service.caller, &path).await?)
}

/// A rollout's own gym run. `kind: "rl"` is the caller's to set and this does
/// not set it — a batch that wants to stay off the evaluation board says so, and
/// a service silently rewriting the body it forwards is worse than one that does
/// not.
#[post("/gym/runs")]
async fn create_run(
    service: ServiceCaller,
    body: web::Json<Value>,
    gyms: web::Data<gyms_engine::Proxy>,
) -> Answer {
    answer(
        gyms.post(&service.caller, "/gym/runs", body.into_inner())
            .await?,
    )
}

/// One move of an engine-driven episode: run the call, record it, and answer
/// with what the model should read.
#[post("/gym/episodes/{id}/step")]
async fn step(
    service: ServiceCaller,
    id: web::Path<String>,
    body: web::Json<Value>,
    gyms: web::Data<gyms_engine::Proxy>,
) -> Answer {
    let body = body.into_inner();
    let budget = tools::Config::from_env().bash_budget_chars;
    answer(
        gym::play(
            &gyms,
            &service.caller,
            &id,
            body["tool"].as_str().unwrap_or_default(),
            body["argument"].as_str().unwrap_or_default(),
            body["reply"].as_str().unwrap_or_default(),
            budget,
        )
        .await?,
    )
}

#[post("/gym/episodes/{id}/finish")]
async fn finish(
    service: ServiceCaller,
    id: web::Path<String>,
    body: web::Json<Value>,
    gyms: web::Data<gyms_engine::Proxy>,
) -> Answer {
    let path = format!("/gym/episodes/{id}/finish");
    answer(gyms.post(&service.caller, &path, body.into_inner()).await?)
}

#[post("/gym/episodes/{id}/close")]
async fn close(
    service: ServiceCaller,
    id: web::Path<String>,
    gyms: web::Data<gyms_engine::Proxy>,
) -> Answer {
    let path = format!("/gym/episodes/{id}/close");
    answer(gyms.post(&service.caller, &path, Value::Null).await?)
}

#[post("/gym/runs/{id}/status")]
async fn set_status(
    service: ServiceCaller,
    id: web::Path<String>,
    body: web::Json<Value>,
    gyms: web::Data<gyms_engine::Proxy>,
) -> Answer {
    let path = format!("/gym/runs/{id}/status");
    answer(gyms.post(&service.caller, &path, body.into_inner()).await?)
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(catalogue);
    cfg.service(dataset);
    cfg.service(documents);
    cfg.service(create_run);
    cfg.service(step);
    cfg.service(finish);
    cfg.service(close);
    cfg.service(set_status);
}

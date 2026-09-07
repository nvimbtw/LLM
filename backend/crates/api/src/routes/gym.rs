//! The gyms service, reached through here.
//!
//! Pass-through, like `data.rs` and `models.rs`: the shape of a run belongs to
//! the service that owns it. The one thing this side adds is the gate — every
//! route is `AdminUser`, because creating a run seeds sandboxes and driving one
//! spends the GPU the chat queue is sharing.
//!
//! The socket that actually drives a run is not here: it is
//! `routes/stream.rs::gym_socket`, on the `/ws` scope with the other two.

use actix_web::{HttpRequest, HttpResponse, get, post, web};
use serde_json::{Value, json};

use crate::demonstrations;
use crate::extractors::AdminUser;
use crate::upstream::{Answer, admin_caller, answer, with_query};

use gyms::Proxy;

#[get("")]
async fn catalogue(admin: AdminUser, gyms: web::Data<Proxy>) -> Answer {
    answer(gyms.get(&admin_caller(&admin), "/gyms").await?)
}

#[post("/runs")]
async fn create_run(admin: AdminUser, body: web::Json<Value>, gyms: web::Data<Proxy>) -> Answer {
    answer(gyms.post(&admin_caller(&admin), "/gym/runs", body.into_inner()).await?)
}

/// The board. `?kind=rl` reads back a training run's batches instead, which is
/// the only way to see them: the default listing is evaluations, because a
/// training run creates one gym run per optimizer step and would otherwise be
/// the whole of it.
#[get("/runs")]
async fn runs(admin: AdminUser, req: HttpRequest, gyms: web::Data<Proxy>) -> Answer {
    let path = with_query("/gym/runs", &req);
    answer(gyms.get(&admin_caller(&admin), &path).await?)
}

#[get("/runs/{id}")]
async fn run(admin: AdminUser, id: web::Path<String>, gyms: web::Data<Proxy>) -> Answer {
    answer(gyms.get(&admin_caller(&admin), &format!("/gym/runs/{id}")).await?)
}

#[get("/episodes/{id}")]
async fn episode(admin: AdminUser, id: web::Path<String>, gyms: web::Data<Proxy>) -> Answer {
    answer(gyms.get(&admin_caller(&admin), &format!("/gym/episodes/{id}")).await?)
}

/// A command a person typed, rather than one a model called. Kept because a
/// task that no model can solve is a task to look at by hand, and this is the
/// only way to look.
#[post("/episodes/{id}/step")]
async fn step(
    admin: AdminUser,
    id: web::Path<String>,
    body: web::Json<Value>,
    gyms: web::Data<Proxy>,
) -> Answer {
    let path = format!("/gym/episodes/{id}/step");
    answer(gyms.post(&admin_caller(&admin), &path, body.into_inner()).await?)
}

#[post("/episodes/{id}/close")]
async fn close(admin: AdminUser, id: web::Path<String>, gyms: web::Data<Proxy>) -> Answer {
    let path = format!("/gym/episodes/{id}/close");
    answer(gyms.post(&admin_caller(&admin), &path, Value::Null).await?)
}

/// Generate expert demonstrations and store them as a dataset to fine-tune on.
///
/// The one route here that is not a pass-through, and it cannot be: the gyms
/// service plays the trajectory, this crate renders what the model reads back,
/// and the data engine owns every corpus. Three services, and the two Python
/// ones never speak — so the assembly is here, where all three are reachable.
/// See `crate::demonstrations`.
///
/// Synchronous rather than a queued run. A batch is bounded by the gyms
/// service's own ceiling on trajectories per request, and the expensive part is
/// the sandbox, not this. A caller wanting more asks again into the same name,
/// which appends.
#[post("/demonstrations")]
async fn demonstrate(
    admin: AdminUser,
    body: web::Json<Value>,
    gyms: web::Data<Proxy>,
    data: web::Data<data_engine::Proxy>,
) -> Answer {
    let caller = admin_caller(&admin);
    let body = body.into_inner();

    let gym_id = body["gym_id"].as_str().unwrap_or("bash").to_string();
    let task_types = body["task_types"].clone();
    let seed_base = body["seed_base"].as_i64().unwrap_or_default();
    // The same budget a live round is rendered at, read from the same place, so
    // a demonstration cannot be clamped differently from the thing it teaches.
    let budget = tools::Config::from_env().bash_budget_chars;

    let played = gyms
        .post(
            &caller,
            "/gym/trajectories",
            json!({
                "gym_id": gym_id,
                "task_types": task_types,
                "count": body["count"].clone(),
                "seed_base": seed_base,
            }),
        )
        .await?;

    let conversations: Vec<Value> = played["trajectories"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|trajectory| demonstrations::conversation(trajectory, budget))
        .collect();

    // Nothing trainable came back. Refused rather than stored, because an empty
    // append would answer 201 with a set that grew by zero — which reads as
    // success and is the same shape as a batch that worked.
    if conversations.is_empty() {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "the gym produced no trainable demonstrations",
        })));
    }

    let stored = data
        .post(
            &caller,
            "/datasets/demonstrations",
            json!({
                "name": body["dataset_name"].clone(),
                "description": body.get("description").cloned().unwrap_or(json!("")),
                "steps": demonstrations::provenance(&gym_id, &task_types, seed_base, budget),
                "conversations": conversations,
            }),
        )
        .await?;

    answer(stored)
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(catalogue);
    cfg.service(create_run);
    cfg.service(runs);
    cfg.service(run);
    cfg.service(episode);
    cfg.service(step);
    cfg.service(close);
    cfg.service(demonstrate);
}

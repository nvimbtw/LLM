// Post-training cycles over HTTP: start one, watch it, pause, resume, stop.
//
// Thin on purpose — every decision a cycle makes belongs to the driver in
// `cycle.rs`, which is the only thing that writes a stage. These handlers start
// one, read one, and change its status; nothing here advances a run.
//
// Admin-only, like /jobs: a cycle queues hours of GPU work.

use actix_web::{HttpResponse, Responder, delete, get, post, web};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::sync::Arc;
use uuid::Uuid;

use crate::cycle::{CycleConfig, Cycles};
use crate::db;
use crate::db::cycle::{CycleStage, CycleStatus};
use crate::extractors::AdminUser;
use storage;

const HISTORY_LIMIT: i64 = 50;

#[derive(Serialize)]
struct CycleResponse {
    id_cycle: Uuid,
    label: String,
    status: CycleStatus,
    stage: CycleStage,
    round: i32,
    max_rounds: i32,
    id_model_start: Option<Uuid>,
    id_model_current: Option<Uuid>,
    id_model_best: Option<Uuid>,
    best_score: Option<f64>,
    id_job_current: Option<Uuid>,
    config: serde_json::Value,
    halt_reason: Option<String>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    finished_at: Option<DateTime<Utc>>,
}

impl From<db::cycle::Cycle> for CycleResponse {
    fn from(c: db::cycle::Cycle) -> Self {
        CycleResponse {
            id_cycle: c.id_cycle,
            label: c.label,
            status: c.status,
            stage: c.stage,
            round: c.round,
            max_rounds: c.max_rounds,
            id_model_start: c.id_model_start,
            id_model_current: c.id_model_current,
            id_model_best: c.id_model_best,
            best_score: c.best_score,
            id_job_current: c.id_job_current,
            config: c.config,
            halt_reason: c.halt_reason,
            created_at: c.created_at,
            updated_at: c.updated_at,
            finished_at: c.finished_at,
        }
    }
}

#[derive(Deserialize)]
struct CreatePayload {
    // The model the first round trains from.
    id_model: Uuid,
    label: Option<String>,
    #[serde(default = "default_max_rounds")]
    max_rounds: i32,
    #[serde(default)]
    config: Option<CycleConfig>,
}

// Enough rounds to see a trend and find a plateau, few enough that an unattended
// run is a night rather than a week. The loop usually halts on its own first.
fn default_max_rounds() -> i32 {
    5
}

#[post("")]
async fn create_cycle(
    admin: AdminUser,
    payload: web::Json<CreatePayload>,
    pool: web::Data<PgPool>,
    cycles: web::Data<Arc<Cycles>>,
) -> impl Responder {
    let CreatePayload {
        id_model,
        label,
        max_rounds,
        config,
    } = payload.into_inner();

    if !(1..=50).contains(&max_rounds) {
        return HttpResponse::BadRequest().body("max_rounds must be between 1 and 50");
    }

    let model = match db::model::find_by_id(pool.get_ref(), id_model).await {
        Ok(Some(model)) => model,
        Ok(None) => return HttpResponse::NotFound().body("model not found"),
        Err(e) => {
            eprintln!("cycle model lookup db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to start the cycle");
        }
    };

    // Checked now rather than three stages in: every round generates from this
    // model, and a model that cannot generate cannot start a cycle.
    let model_dir = storage::models_dir().join(id_model.to_string());
    if !model_dir.join("merges.bin").exists() {
        return HttpResponse::BadRequest()
            .body("model has no tokenizer on disk; it cannot generate");
    }

    // A cycle's first stage calibrates against the human ratings, so the seed bar
    // has to be cleared before it starts — otherwise it queues, runs, and halts on
    // its first job for a reason that was knowable up front.
    let counts = match db::signal::counts(pool.get_ref()).await {
        Ok(counts) => counts,
        Err(e) => {
            eprintln!("cycle signal count db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to start the cycle");
        }
    };
    if let Err(msg) = crate::gather::check_signals(counts.desirable, counts.undesirable) {
        return HttpResponse::BadRequest().body(msg);
    }

    let config = config.unwrap_or_default();
    let config = match serde_json::to_value(&config) {
        Ok(value) => value,
        Err(e) => {
            eprintln!("serialize cycle config error: {}", e);
            return HttpResponse::InternalServerError().body("failed to start the cycle");
        }
    };
    let label = label.unwrap_or_else(|| model.name.clone());

    match db::cycle::create(
        pool.get_ref(),
        &label,
        id_model,
        max_rounds,
        &config,
        Some(admin.claims.sub),
    )
    .await
    {
        Ok(cycle) => {
            cycles.wake();
            HttpResponse::Created().json(CycleResponse::from(cycle))
        }
        Err(e) => {
            eprintln!("create cycle db error: {}", e);
            HttpResponse::InternalServerError().body("failed to start the cycle")
        }
    }
}

#[get("")]
async fn list_cycles(_admin: AdminUser, pool: web::Data<PgPool>) -> impl Responder {
    match db::cycle::list_all(pool.get_ref(), HISTORY_LIMIT).await {
        Ok(cycles) => HttpResponse::Ok().json(
            cycles
                .into_iter()
                .map(CycleResponse::from)
                .collect::<Vec<_>>(),
        ),
        Err(e) => {
            eprintln!("list cycles db error: {}", e);
            HttpResponse::InternalServerError().body("failed to load cycles")
        }
    }
}

// One cycle with its rounds and the jobs it has spawned — everything the cycle
// page renders, in one request. Three small queries rather than three round trips.
#[derive(Serialize)]
struct CycleDetail {
    #[serde(flatten)]
    cycle: CycleResponse,
    rounds: Vec<db::cycle::Round>,
    jobs: Vec<CycleJobResponse>,
}

#[derive(Serialize)]
struct CycleJobResponse {
    id_job: Uuid,
    kind: db::job::JobKind,
    status: db::job::JobStatus,
    label: String,
    cycle_role: Option<String>,
    created_at: DateTime<Utc>,
}

#[get("/{id}")]
async fn get_cycle(
    _admin: AdminUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let id_cycle = path.into_inner();
    let cycle = match db::cycle::find_by_id(pool.get_ref(), id_cycle).await {
        Ok(Some(cycle)) => cycle,
        Ok(None) => return HttpResponse::NotFound().body("cycle not found"),
        Err(e) => {
            eprintln!("get cycle db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to load the cycle");
        }
    };

    let rounds = match db::cycle::list_rounds(pool.get_ref(), id_cycle).await {
        Ok(rounds) => rounds,
        Err(e) => {
            eprintln!("list rounds db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to load the cycle");
        }
    };
    let jobs = match db::cycle::list_jobs(pool.get_ref(), id_cycle).await {
        Ok(jobs) => jobs
            .into_iter()
            .map(|j| CycleJobResponse {
                id_job: j.id_job,
                kind: j.kind,
                status: j.status,
                label: j.label,
                cycle_role: j.cycle_role,
                created_at: j.created_at,
            })
            .collect(),
        Err(e) => {
            eprintln!("list cycle jobs db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to load the cycle");
        }
    };

    HttpResponse::Ok().json(CycleDetail {
        cycle: CycleResponse::from(cycle),
        rounds,
        jobs,
    })
}

#[derive(Deserialize)]
struct StatusPayload {
    // "running" to resume, "paused" to hold, "halted" to stop for good.
    status: CycleStatusInput,
}

#[derive(Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum CycleStatusInput {
    Running,
    Paused,
    Halted,
}

// Pause, resume, or stop. Deliberately the only mutation: a cycle's *stage* is the
// driver's alone, so there is no endpoint that skips a step or re-runs one — those
// would race the driver for the same row.
#[post("/{id}/status")]
async fn set_cycle_status(
    _admin: AdminUser,
    path: web::Path<Uuid>,
    payload: web::Json<StatusPayload>,
    pool: web::Data<PgPool>,
    cycles: web::Data<Arc<Cycles>>,
) -> impl Responder {
    let id_cycle = path.into_inner();
    let cycle = match db::cycle::find_by_id(pool.get_ref(), id_cycle).await {
        Ok(Some(cycle)) => cycle,
        Ok(None) => return HttpResponse::NotFound().body("cycle not found"),
        Err(e) => {
            eprintln!("get cycle db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to update the cycle");
        }
    };
    if cycle.status.is_terminal() {
        return HttpResponse::BadRequest().body("this cycle has already finished");
    }

    let (status, reason) = match payload.status {
        CycleStatusInput::Running => (CycleStatus::Running, None),
        CycleStatusInput::Paused => (CycleStatus::Paused, Some("paused")),
        CycleStatusInput::Halted => (CycleStatus::Halted, Some("stopped by hand")),
    };

    match db::cycle::set_status(pool.get_ref(), id_cycle, status, reason).await {
        Ok(()) => {
            // Resuming has to nudge the driver: it is edge-triggered, and nothing
            // else is about to happen on a cycle whose jobs have all finished.
            cycles.wake();
            HttpResponse::NoContent().finish()
        }
        Err(e) => {
            eprintln!("set cycle status db error: {}", e);
            HttpResponse::InternalServerError().body("failed to update the cycle")
        }
    }
}

#[delete("/{id}")]
async fn delete_cycle(
    _admin: AdminUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let id_cycle = path.into_inner();
    let cycle = match db::cycle::find_by_id(pool.get_ref(), id_cycle).await {
        Ok(Some(cycle)) => cycle,
        Ok(None) => return HttpResponse::NotFound().body("cycle not found"),
        Err(e) => {
            eprintln!("get cycle db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to delete the cycle");
        }
    };
    // A running cycle would be re-advanced by the driver mid-delete, and its jobs
    // would carry on with nothing to report to.
    if !cycle.status.is_terminal() {
        return HttpResponse::BadRequest().body("stop the cycle before deleting it");
    }

    match db::cycle::delete(pool.get_ref(), id_cycle).await {
        Ok(true) => HttpResponse::NoContent().finish(),
        Ok(false) => HttpResponse::NotFound().body("cycle not found"),
        Err(e) => {
            eprintln!("delete cycle db error: {}", e);
            HttpResponse::InternalServerError().body("failed to delete the cycle")
        }
    }
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(list_cycles);
    cfg.service(create_cycle);
    cfg.service(get_cycle);
    cfg.service(set_cycle_status);
    cfg.service(delete_cycle);
}

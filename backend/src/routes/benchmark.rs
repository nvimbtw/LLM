// Benchmark history and analysis.
//
// Starting a benchmark moved to `POST /jobs` — a run is a `benchmark` job, and the
// queue is what guarantees it isn't fighting a training run for the device. That
// removed the two `is_busy`-style cross-checks this file used to carry: the
// invariant is now structural rather than something each route has to remember to
// assert.
//
// A run's row is opened by the runner when it reaches the GPU (not at enqueue), so
// everything here is history: past scores, the trend, and the suite's state.

use actix_web::{HttpResponse, Responder, delete, get, web};
use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

use crate::db;
use crate::db::benchmark::BenchmarkStatus;
use crate::extractors::AdminUser;
use crate::storage;

// How many runs the history list returns. The page charts a trend over past runs;
// a couple of hundred is far more than that needs and keeps the response small.
const HISTORY_LIMIT: i64 = 200;

// A row in the history list. Live progress is deliberately absent: it belongs to
// the job that is producing the run, and `/jobs` is where in-flight work is shown.
#[derive(Serialize)]
struct RunSummary {
    id_run: Uuid,
    id_model: Uuid,
    model_name: String,
    status: BenchmarkStatus,
    score: Option<f64>,
    error: Option<String>,
    // Tokens the model was trained on. Drives the scaling chart (score against
    // training tokens); null for a model whose config never recorded a corpus size.
    train_tokens: Option<i64>,
    created_at: DateTime<Utc>,
    completed_at: Option<DateTime<Utc>>,
}

impl From<db::benchmark::BenchmarkSummary> for RunSummary {
    fn from(run: db::benchmark::BenchmarkSummary) -> Self {
        RunSummary {
            id_run: run.id_run,
            id_model: run.id_model,
            model_name: run.model_name,
            status: run.status,
            score: run.score,
            error: run.error,
            train_tokens: run.train_tokens,
            created_at: run.created_at,
            completed_at: run.completed_at,
        }
    }
}

// A single run with its full report.
#[derive(Serialize)]
struct RunDetail {
    id_run: Uuid,
    id_model: Uuid,
    model_name: String,
    status: BenchmarkStatus,
    score: Option<f64>,
    report: Option<Value>,
    error: Option<String>,
    train_tokens: Option<i64>,
    created_at: DateTime<Utc>,
    completed_at: Option<DateTime<Utc>>,
}

// Every run, newest first — this is the "previous scores" history the page charts.
#[get("/runs")]
async fn list_runs(_admin: AdminUser, pool: web::Data<PgPool>) -> impl Responder {
    match db::benchmark::list(pool.get_ref(), HISTORY_LIMIT).await {
        Ok(runs) => {
            let body: Vec<RunSummary> = runs.into_iter().map(RunSummary::from).collect();
            HttpResponse::Ok().json(body)
        }
        Err(e) => {
            eprintln!("list benchmark runs db error: {}", e);
            HttpResponse::InternalServerError().body("failed to list benchmark runs")
        }
    }
}

// One model's benchmark record: its latest full report plus every run of it, for
// the results section of the model page.
//
// One response rather than two requests because the section renders as one thing —
// a headline score with the runs behind it — and two round trips would let it paint
// with a score and no history, or history and no score.
#[derive(Serialize)]
struct ModelRuns {
    latest: Option<RunDetail>,
    runs: Vec<RunSummary>,
}

// How many of one model's runs to return. A model is benchmarked a handful of times
// at most; this is a bound, not a page.
const MODEL_RUNS_LIMIT: i64 = 50;

#[get("/models/{id}/runs")]
async fn list_model_runs(
    _admin: AdminUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let id_model = path.into_inner();
    let runs = match db::benchmark::list_for_model(pool.get_ref(), id_model, MODEL_RUNS_LIMIT).await
    {
        Ok(runs) => runs,
        Err(e) => {
            eprintln!("list model benchmark runs db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to list benchmark runs");
        }
    };
    let latest = match db::benchmark::latest_report_for_model(pool.get_ref(), id_model).await {
        Ok(run) => run.map(|run| RunDetail {
            id_run: run.id_run,
            id_model: run.id_model,
            model_name: run.model_name,
            status: run.status,
            score: run.score,
            report: run.report,
            error: run.error,
            train_tokens: run.train_tokens,
            created_at: run.created_at,
            completed_at: run.completed_at,
        }),
        Err(e) => {
            eprintln!("latest model benchmark report db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to load benchmark run");
        }
    };

    HttpResponse::Ok().json(ModelRuns {
        latest,
        runs: runs.into_iter().map(RunSummary::from).collect(),
    })
}

#[get("/runs/{id}")]
async fn get_run(
    _admin: AdminUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    match db::benchmark::find_by_id(pool.get_ref(), path.into_inner()).await {
        Ok(Some(run)) => HttpResponse::Ok().json(RunDetail {
            id_run: run.id_run,
            id_model: run.id_model,
            model_name: run.model_name,
            status: run.status,
            score: run.score,
            report: run.report,
            error: run.error,
            train_tokens: run.train_tokens,
            created_at: run.created_at,
            completed_at: run.completed_at,
        }),
        Ok(None) => HttpResponse::NotFound().body("benchmark run not found"),
        Err(e) => {
            eprintln!("get benchmark run db error: {}", e);
            HttpResponse::InternalServerError().body("failed to load benchmark run")
        }
    }
}

#[delete("/runs/{id}")]
async fn delete_run(
    _admin: AdminUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    let id_run = path.into_inner();
    // Deleting the row of an in-flight run would leave its worker with nothing to
    // write to, and the completion would then be silently dropped. `running` is the
    // durable marker for that, so no in-memory registry needs consulting.
    match db::benchmark::find_by_id(pool.get_ref(), id_run).await {
        Ok(Some(run)) if run.status == BenchmarkStatus::Running => {
            return HttpResponse::Conflict().body("this run is still in progress");
        }
        Ok(Some(_)) => {}
        Ok(None) => return HttpResponse::NotFound().body("benchmark run not found"),
        Err(e) => {
            eprintln!("delete benchmark run lookup db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to delete benchmark run");
        }
    }

    match db::benchmark::delete(pool.get_ref(), id_run).await {
        Ok(true) => HttpResponse::NoContent().finish(),
        Ok(false) => HttpResponse::NotFound().body("benchmark run not found"),
        Err(e) => {
            eprintln!("delete benchmark run db error: {}", e);
            HttpResponse::InternalServerError().body("failed to delete benchmark run")
        }
    }
}

// Whether the reference corpus is installed, so the page can tell the admin *why*
// the perplexity section will be empty before they spend a run finding out.
#[derive(Serialize)]
struct SuiteInfo {
    corpus_dir: String,
    corpus_files: usize,
    blimp_dir: String,
    blimp_files: usize,
    chat_dir: String,
    chat_files: usize,
}

fn count_files(dir: &std::path::Path, extension: &str) -> usize {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some(extension))
        .count()
}

#[get("/suite")]
async fn get_suite(_admin: AdminUser) -> impl Responder {
    let eval_dir = storage::eval_dir();
    let corpus_dir = eval_dir.join("corpus");
    let blimp_dir = eval_dir.join("blimp");
    let chat_dir = eval_dir.join("chat");
    HttpResponse::Ok().json(SuiteInfo {
        corpus_files: count_files(&corpus_dir, "txt"),
        blimp_files: count_files(&blimp_dir, "jsonl"),
        chat_files: count_files(&chat_dir, "jsonl"),
        corpus_dir: corpus_dir.to_string_lossy().to_string(),
        blimp_dir: blimp_dir.to_string_lossy().to_string(),
        chat_dir: chat_dir.to_string_lossy().to_string(),
    })
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(list_runs);
    // Registered before `get_run` (GET /runs/{id}) for the same reason the jobs
    // scope orders its literal paths first — a distinct prefix here, but keeping the
    // specific-before-generic order is what stops the next literal route from being
    // swallowed silently.
    cfg.service(list_model_runs);
    cfg.service(get_run);
    cfg.service(delete_run);
    cfg.service(get_suite);
}

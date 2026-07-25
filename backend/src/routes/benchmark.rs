use actix_web::{HttpResponse, Responder, delete, get, post, web};
use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use sqlx::PgPool;
use uuid::Uuid;

use crate::benchmarks::{BenchmarkRegistry, Progress};
use crate::db;
use crate::db::benchmark::BenchmarkStatus;
use crate::extractors::AdminUser;
use crate::jobs::JobRegistry;
use crate::storage;

// How many runs the history list returns. The page charts a trend over past runs;
// a couple of hundred is far more than that needs and keeps the response small.
const HISTORY_LIMIT: i64 = 200;

// A row in the history list. `progress` is present only while a run is in flight —
// it lives in the in-memory registry, not the database.
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
    progress: Option<Progress>,
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
    progress: Option<Progress>,
}

#[derive(serde::Deserialize)]
struct StartPayload {
    id_model: Uuid,
}

// Starts a benchmark for one model.
//
// Admin-only, and not because the results are sensitive: a run monopolizes the GPU
// for tens of seconds, which would let any logged-in user stall training.
#[post("")]
async fn start_benchmark(
    _admin: AdminUser,
    payload: web::Json<StartPayload>,
    pool: web::Data<PgPool>,
    registry: web::Data<BenchmarkRegistry>,
    jobs: web::Data<JobRegistry>,
) -> impl Responder {
    let id_model = payload.id_model;

    // One GPU, one job. Two benchmarks — or a benchmark against a live training
    // run — contend for the device and surface as an opaque CUDA fault rather than
    // a legible error, so both cases are refused up front.
    if registry.is_busy() {
        return HttpResponse::Conflict().body("a benchmark is already running");
    }
    if jobs.list().iter().any(|(_, status, _)| !status.is_terminal()) {
        return HttpResponse::Conflict()
            .body("a training run is in progress; benchmarks share the GPU");
    }

    let model = match db::model::find_by_id(pool.get_ref(), id_model).await {
        Ok(Some(model)) => model,
        Ok(None) => return HttpResponse::NotFound().body("model not found"),
        Err(e) => {
            eprintln!("benchmark model lookup db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to start benchmark");
        }
    };

    // Weights are keyed by the model's UUID, not its name, so a renamed model
    // still resolves. Checked here rather than in the worker so a missing
    // directory is a 400 the client can act on, not a failed run row.
    let model_dir = storage::models_dir().join(id_model.to_string());
    if !model_dir.join("merges.bin").exists() {
        return HttpResponse::BadRequest()
            .body("model has no tokenizer on disk; it cannot be benchmarked");
    }

    let id_run = Uuid::new_v4();
    let train_tokens = db::model::train_tokens(&model);
    match db::benchmark::create(pool.get_ref(), id_run, id_model, &model.name, train_tokens).await {
        Ok(run) => {
            registry.start(id_run, model_dir);
            HttpResponse::Accepted().json(RunSummary {
                id_run: run.id_run,
                id_model: run.id_model,
                model_name: run.model_name,
                status: run.status,
                score: run.score,
                error: run.error,
                train_tokens: run.train_tokens,
                created_at: run.created_at,
                completed_at: run.completed_at,
                progress: registry.progress(id_run),
            })
        }
        Err(e) => {
            eprintln!("create benchmark run db error: {}", e);
            HttpResponse::InternalServerError().body("failed to start benchmark")
        }
    }
}

// Every run, newest first — this is the "previous scores" history the page charts.
#[get("/runs")]
async fn list_runs(
    _admin: AdminUser,
    pool: web::Data<PgPool>,
    registry: web::Data<BenchmarkRegistry>,
) -> impl Responder {
    match db::benchmark::list(pool.get_ref(), HISTORY_LIMIT).await {
        Ok(runs) => {
            let body: Vec<RunSummary> = runs
                .into_iter()
                .map(|run| RunSummary {
                    progress: match run.status {
                        BenchmarkStatus::Running => registry.progress(run.id_run),
                        _ => None,
                    },
                    id_run: run.id_run,
                    id_model: run.id_model,
                    model_name: run.model_name,
                    status: run.status,
                    score: run.score,
                    error: run.error,
                    train_tokens: run.train_tokens,
                    created_at: run.created_at,
                    completed_at: run.completed_at,
                })
                .collect();
            HttpResponse::Ok().json(body)
        }
        Err(e) => {
            eprintln!("list benchmark runs db error: {}", e);
            HttpResponse::InternalServerError().body("failed to list benchmark runs")
        }
    }
}

#[get("/runs/{id}")]
async fn get_run(
    _admin: AdminUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
    registry: web::Data<BenchmarkRegistry>,
) -> impl Responder {
    let id_run = path.into_inner();
    match db::benchmark::find_by_id(pool.get_ref(), id_run).await {
        Ok(Some(run)) => HttpResponse::Ok().json(RunDetail {
            progress: match run.status {
                BenchmarkStatus::Running => registry.progress(id_run),
                _ => None,
            },
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
    registry: web::Data<BenchmarkRegistry>,
) -> impl Responder {
    let id_run = path.into_inner();
    // Deleting the row of an in-flight run would leave its worker with nothing to
    // write to, and the completion would then be silently dropped.
    if registry.progress(id_run).is_some() {
        return HttpResponse::Conflict().body("this run is still in progress");
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
    HttpResponse::Ok().json(SuiteInfo {
        corpus_files: count_files(&corpus_dir, "txt"),
        blimp_files: count_files(&blimp_dir, "jsonl"),
        corpus_dir: corpus_dir.to_string_lossy().to_string(),
        blimp_dir: blimp_dir.to_string_lossy().to_string(),
    })
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(start_benchmark);
    cfg.service(list_runs);
    cfg.service(get_run);
    cfg.service(delete_run);
    cfg.service(get_suite);
}

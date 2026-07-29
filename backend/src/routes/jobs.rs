#![allow(non_snake_case)]

// The queue's HTTP surface. Everything that touches the GPU is a row in `jobs`, so
// this is where work is queued, watched, reordered, cancelled and deleted.
//
// Admin-only throughout, and not because the results are sensitive: a job
// monopolizes the one GPU, so queueing is the ability to stall everyone else's
// work. The exception is `chat`, which any logged-in user creates — but through
// the chat socket, which enqueues server-side, never through `POST /jobs`.

use actix_web::{HttpRequest, HttpResponse, Responder, delete, get, patch, post, web};
use chrono::{DateTime, Utc};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::PgPool;
use std::sync::Arc;
use uuid::Uuid;

use db;
use db::job::{Job, JobKind, JobLane, JobStatus};
use crate::extractors::AdminUser;
use crate::queue::Queue;
use storage;
use crate::training::{TrainConfig, TrainMode};
use crate::ws;

// One page of history, when the client doesn't say. Sized for a screenful rather
// than for the archive: the board refetches the current page on every queue nudge,
// so the page is the unit that has to stay cheap.
const HISTORY_PAGE: i64 = 25;
// The most a client may ask for in one page, so a hand-written `history_limit` can't
// pull the entire jobs table (config JSONB and all) into one response.
const HISTORY_PAGE_MAX: i64 = 200;

// One job as the frontend sees it. `config` is included so the retry flow can seed
// a composer from a finished job without a second request.
#[derive(Serialize)]
struct JobResponse {
    id_job: Uuid,
    kind: JobKind,
    // Which runner this row belongs to. Surfaced so the board can show the two
    // lines as the separate queues they are, rather than one list where a chat turn
    // appears to be waiting behind a generation run it will in fact overtake.
    lane: JobLane,
    // Background jobs yield to interactive work. Surfaced so the board can say why
    // a queued job is not running even though the device looks free.
    background: bool,
    status: JobStatus,
    label: String,
    config: Value,
    progress: f64,
    stage: Option<String>,
    ref_id: Option<Uuid>,
    error: Option<String>,
    created_at: DateTime<Utc>,
    started_at: Option<DateTime<Utc>>,
    finished_at: Option<DateTime<Utc>>,
}

impl From<Job> for JobResponse {
    fn from(job: Job) -> Self {
        JobResponse {
            id_job: job.id_job,
            kind: job.kind,
            lane: job.lane,
            background: job.background,
            status: job.status,
            label: job.label,
            config: job.config,
            progress: job.progress,
            stage: job.stage,
            ref_id: job.ref_id,
            error: job.error,
            created_at: job.created_at,
            started_at: job.started_at,
            finished_at: job.finished_at,
        }
    }
}

// The whole board in one response: the live band in execution order, the finished
// jobs newest-first, and whether the runner is draining. One request rather than
// three, because the page renders them as a single ledger and three round trips
// would let it paint in inconsistent states.
#[derive(Serialize)]
struct BoardResponse {
    active: Vec<JobResponse>,
    history: Vec<JobResponse>,
    // Which slice of the history this is, and how many finished jobs exist — enough
    // for the client to draw a pager without a second request or a guess about
    // whether it is on the last page.
    history_offset: i64,
    history_limit: i64,
    history_total: i64,
    paused: bool,
}

// Which page of history to return. Both fields are optional: a client that says
// nothing gets the first page, which is what the board wants on load.
#[derive(Deserialize)]
struct BoardQuery {
    #[serde(default)]
    history_offset: Option<i64>,
    #[serde(default)]
    history_limit: Option<i64>,
}

#[get("")]
async fn list_jobs(
    _admin: AdminUser,
    pool: web::Data<PgPool>,
    query: web::Query<BoardQuery>,
) -> impl Responder {
    // Clamped rather than rejected: a nonsensical page is a client bug, and the
    // useful response to it is the first page, not a 400 that blanks the board.
    let limit = query
        .history_limit
        .unwrap_or(HISTORY_PAGE)
        .clamp(1, HISTORY_PAGE_MAX);
    let offset = query.history_offset.unwrap_or(0).max(0);

    let active = match db::job::list_active(pool.get_ref()).await {
        Ok(jobs) => jobs,
        Err(e) => {
            eprintln!("list active jobs db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to list jobs");
        }
    };
    let history = match db::job::list_history(pool.get_ref(), limit, offset).await {
        Ok(jobs) => jobs,
        Err(e) => {
            eprintln!("list job history db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to list jobs");
        }
    };
    let history_total = match db::job::count_history(pool.get_ref()).await {
        Ok(total) => total,
        Err(e) => {
            eprintln!("count job history db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to list jobs");
        }
    };
    let paused = match db::job::is_paused(pool.get_ref()).await {
        Ok(paused) => paused,
        Err(e) => {
            eprintln!("read pause state db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to list jobs");
        }
    };

    HttpResponse::Ok().json(BoardResponse {
        active: active.into_iter().map(JobResponse::from).collect(),
        history: history.into_iter().map(JobResponse::from).collect(),
        history_offset: offset,
        history_limit: limit,
        history_total,
        paused,
    })
}

#[get("/{id}")]
async fn get_job(
    _admin: AdminUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
) -> impl Responder {
    match db::job::find_by_id(pool.get_ref(), path.into_inner()).await {
        Ok(Some(job)) => HttpResponse::Ok().json(JobResponse::from(job)),
        Ok(None) => HttpResponse::NotFound().body("job not found"),
        Err(e) => {
            eprintln!("get job db error: {}", e);
            HttpResponse::InternalServerError().body("failed to load job")
        }
    }
}

// A new job. `config` is validated against the kind here rather than in the runner,
// so a bad training config is a 400 the moment you submit it instead of a failed
// row hours later when it reaches the front of the queue.
#[derive(Deserialize)]
struct CreatePayload {
    kind: JobKind,
    config: Value,
}

#[post("")]
async fn create_job(
    admin: AdminUser,
    payload: web::Json<CreatePayload>,
    pool: web::Data<PgPool>,
    queue: web::Data<Arc<Queue>>,
) -> impl Responder {
    let CreatePayload { kind, config } = payload.into_inner();

    let (label, config) = match kind {
        JobKind::Train => match prepare_train(pool.get_ref(), config).await {
            Ok(prepared) => prepared,
            Err(response) => return response,
        },
        JobKind::Benchmark => match prepare_benchmark(pool.get_ref(), config).await {
            Ok(prepared) => prepared,
            Err(response) => return response,
        },
        // A chat turn needs a persisted user message to reply to, which only the
        // chat socket can create. Accepting one here would queue a reply to
        // nothing.
        JobKind::Autolabel => match prepare_autolabel(pool.get_ref(), config).await {
            Ok(prepared) => prepared,
            Err(response) => return response,
        },
        JobKind::Gather => match prepare_gather(pool.get_ref(), config).await {
            Ok(prepared) => prepared,
            Err(response) => return response,
        },
        JobKind::Synth => match prepare_synth(pool.get_ref(), config).await {
            Ok(prepared) => prepared,
            Err(response) => return response,
        },
        JobKind::Chat => {
            return HttpResponse::BadRequest().body("chat jobs are queued from the chat socket");
        }
    };

    match db::job::create(
        pool.get_ref(),
        kind,
        &label,
        &config,
        Some(admin.claims.sub),
    )
    .await
    {
        Ok(job) => {
            queue.wake();
            queue.notify_changed();
            HttpResponse::Created().json(JobResponse::from(job))
        }
        Err(e) => {
            eprintln!("create job db error: {}", e);
            HttpResponse::InternalServerError().body("failed to queue job")
        }
    }
}

// Validates a training config and resolves the storage id the run will write to,
// returning the label and the config as it will be stored.
async fn prepare_train(pool: &PgPool, config: Value) -> Result<(String, Value), HttpResponse> {
    let mut config: TrainConfig = match serde_json::from_value(config) {
        Ok(config) => config,
        Err(e) => return Err(HttpResponse::BadRequest().body(format!("invalid training config: {}", e))),
    };
    if let Err(msg) = validate_config(&config) {
        return Err(HttpResponse::BadRequest().body(msg));
    }
    if let Err(msg) = validate_sft(&config) {
        return Err(HttpResponse::BadRequest().body(msg));
    }

    // A fine-tune loads the base model's weights wholesale, so its architecture *is*
    // the base's — the request's own dimensions/context/heads/layers are placeholders
    // the fine-tune composers never collect, and they are what the panels were
    // showing (a 64d/64ctx run that is in fact whatever the base was). The worker
    // inherits them too, but too late for the display: the job's stored config is
    // what every panel reads before the run attaches, and the only thing left once it
    // has finished. Resolving at enqueue makes the row true from the moment it exists.
    //
    // `validate_sft` has already established that a non-pretrain run names a base
    // that resolves. Best-effort past that: a base whose meta.json is unreadable
    // still trains correctly, it just carries the request's numbers.
    if config.mode != TrainMode::Pretrain {
        if let Some(base_id) = config.base_model_id {
            let base_dir = storage::models_dir().join(base_id.to_string());
            if let Err(e) = crate::training::apply_base_architecture(&mut config, &base_dir) {
                eprintln!(
                    "queue {}: could not inherit base architecture ({}); the row keeps the requested values",
                    config.model_name, e
                );
            }
        }
    }

    // A KTO run with too little data used to be discovered the expensive way: the
    // job queued, waited its turn, loaded the base model, and *then* failed in
    // `build_kto_dataset` with "no rated replies found" — or worse, succeeded on
    // thirty examples and quietly narrowed the model.
    //
    // Both bars are checked, seed first: rating is what unblocks auto-labelling,
    // and auto-labelling is what fills the catalogue, so reporting them in that
    // order names the step that is actually next.
    if config.mode == TrainMode::Kto {
        let counts = match db::signal::counts(pool).await {
            Ok(counts) => counts,
            Err(e) => {
                eprintln!("signal count db error: {}", e);
                return Err(HttpResponse::InternalServerError().body("failed to queue job"));
            }
        };
        if let Err(msg) = crate::gather::check_signals(counts.desirable, counts.undesirable) {
            return Err(HttpResponse::BadRequest().body(msg));
        }

        let (auto_up, auto_down) = crate::autolabel::catalogue_counts();
        let (approved, rejected) =
            (counts.desirable + auto_up, counts.undesirable + auto_down);
        if let Err(msg) = crate::gather::check_catalogue(approved, rejected) {
            return Err(HttpResponse::BadRequest().body(msg));
        }

        // The backstop under the catalogue bar. Clamped rather than refused: the
        // config is otherwise fine, and silently running it five times over a
        // small catalogue is the failure — not the request itself.
        let cap = crate::gather::max_epochs_for(approved + rejected);
        if config.epochs > cap {
            config.epochs = cap;
        }
    }

    // RAFT consumes only the approved half, so it is gated on that count alone —
    // and capped on it too, since max-likelihood over a handful of replies for five
    // epochs is the same overfit KTO's cap exists to prevent, with a denser signal
    // driving it.
    if config.mode == TrainMode::Raft {
        let counts = match db::signal::counts(pool).await {
            Ok(counts) => counts,
            Err(e) => {
                eprintln!("signal count db error: {}", e);
                return Err(HttpResponse::InternalServerError().body("failed to queue job"));
            }
        };
        let (auto_up, _) = crate::autolabel::catalogue_counts();
        let approved = counts.desirable + auto_up;
        if let Err(msg) = crate::gather::check_raft(approved) {
            return Err(HttpResponse::BadRequest().body(msg));
        }
        let cap = crate::gather::max_epochs_for(approved);
        if config.epochs > cap {
            config.epochs = cap;
        }
    }

    // Resolve the storage id (which names the on-disk weights dir): reuse the
    // existing model's id on a retrain of the same name, else mint a new one.
    // Done at enqueue so the queued row already says where it will write.
    config.storage_id = match db::model::find_id_by_name(pool, &config.model_name).await {
        Ok(Some(id)) => id,
        Ok(None) => Uuid::new_v4(),
        Err(e) => {
            eprintln!("resolve model id db error: {}", e);
            return Err(HttpResponse::InternalServerError().body("failed to queue job"));
        }
    };

    let label = config.model_name.clone();
    match serde_json::to_value(&config) {
        Ok(value) => Ok((label, value)),
        Err(e) => {
            eprintln!("serialize training config error: {}", e);
            Err(HttpResponse::InternalServerError().body("failed to queue job"))
        }
    }
}

#[derive(Deserialize)]
struct BenchmarkConfig {
    id_model: Uuid,
}

// Checks the model exists and has a tokenizer now, so an impossible benchmark is
// refused at submit rather than occupying a queue slot to fail.
async fn prepare_benchmark(pool: &PgPool, config: Value) -> Result<(String, Value), HttpResponse> {
    let parsed: BenchmarkConfig = match serde_json::from_value(config) {
        Ok(config) => config,
        Err(_) => return Err(HttpResponse::BadRequest().body("a benchmark job needs an id_model")),
    };

    let model = match db::model::find_by_id(pool, parsed.id_model).await {
        Ok(Some(model)) => model,
        Ok(None) => return Err(HttpResponse::NotFound().body("model not found")),
        Err(e) => {
            eprintln!("benchmark model lookup db error: {}", e);
            return Err(HttpResponse::InternalServerError().body("failed to queue job"));
        }
    };

    // Weights are keyed by the model's UUID, not its name, so a renamed model still
    // resolves.
    let model_dir = storage::models_dir().join(parsed.id_model.to_string());
    if !model_dir.join("merges.bin").exists() {
        return Err(HttpResponse::BadRequest()
            .body("model has no tokenizer on disk; it cannot be benchmarked"));
    }

    Ok((
        model.name,
        serde_json::json!({ "id_model": parsed.id_model }),
    ))
}

// Validates an auto-labelling config: the model that will generate the candidates
// must exist and have a tokenizer on disk, and the run must ask for enough
// candidates per prompt to have a best and a worst.
async fn prepare_autolabel(pool: &PgPool, config: Value) -> Result<(String, Value), HttpResponse> {
    let parsed: crate::autolabel::AutolabelRequest = match serde_json::from_value(config) {
        Ok(config) => config,
        Err(_) => {
            return Err(HttpResponse::BadRequest().body("an autolabel job needs an id_model"));
        }
    };

    if parsed.candidates < 2 {
        return Err(HttpResponse::BadRequest()
            .body("candidates must be at least 2 — one reply cannot be ranked against itself"));
    }
    if parsed.max_prompts < 1 {
        return Err(HttpResponse::BadRequest().body("max_prompts must be at least 1"));
    }
    if !(parsed.temperature.is_finite() && parsed.temperature > 0.0) {
        // Temperature 0 is argmax, which would produce N identical candidates and a
        // spread of exactly zero — every prompt would be skipped as flat.
        return Err(HttpResponse::BadRequest().body("temperature must be a positive number"));
    }

    let model = match db::model::find_by_id(pool, parsed.id_model).await {
        Ok(Some(model)) => model,
        Ok(None) => return Err(HttpResponse::NotFound().body("model not found")),
        Err(e) => {
            eprintln!("autolabel model lookup db error: {}", e);
            return Err(HttpResponse::InternalServerError().body("failed to queue job"));
        }
    };

    let model_dir = storage::models_dir().join(parsed.id_model.to_string());
    if !model_dir.join("merges.bin").exists() {
        return Err(HttpResponse::BadRequest()
            .body("model has no tokenizer on disk; it cannot generate"));
    }

    match serde_json::to_value(&parsed) {
        Ok(value) => Ok((model.name, value)),
        Err(e) => {
            eprintln!("serialize autolabel config error: {}", e);
            Err(HttpResponse::InternalServerError().body("failed to queue job"))
        }
    }
}

// Validates a gather config. Nearly `prepare_autolabel` — the same model must
// exist and be able to generate — minus the candidate count, since a gather run
// produces one reply per prompt and there is nothing to rank them against.
async fn prepare_gather(pool: &PgPool, config: Value) -> Result<(String, Value), HttpResponse> {
    let parsed: crate::gather::GatherRequest = match serde_json::from_value(config) {
        Ok(config) => config,
        Err(_) => {
            return Err(HttpResponse::BadRequest().body("a gather job needs an id_model"));
        }
    };

    if parsed.batch < 1 {
        return Err(HttpResponse::BadRequest().body("batch must be at least 1"));
    }
    if !(parsed.temperature.is_finite() && parsed.temperature > 0.0) {
        // Temperature 0 is argmax, which makes every reply to a similar prompt the
        // same reply — a batch of near-identical text to rate.
        return Err(HttpResponse::BadRequest().body("temperature must be a positive number"));
    }

    let model = match db::model::find_by_id(pool, parsed.id_model).await {
        Ok(Some(model)) => model,
        Ok(None) => return Err(HttpResponse::NotFound().body("model not found")),
        Err(e) => {
            eprintln!("gather model lookup db error: {}", e);
            return Err(HttpResponse::InternalServerError().body("failed to queue job"));
        }
    };

    let model_dir = storage::models_dir().join(parsed.id_model.to_string());
    if !model_dir.join("merges.bin").exists() {
        return Err(HttpResponse::BadRequest()
            .body("model has no tokenizer on disk; it cannot generate"));
    }

    match serde_json::to_value(&parsed) {
        Ok(value) => Ok((model.name, value)),
        Err(e) => {
            eprintln!("serialize gather config error: {}", e);
            Err(HttpResponse::InternalServerError().body("failed to queue job"))
        }
    }
}

// Validates a synthetic-generation request.
//
// Everything checkable without spending a model call is checked here: a run that
// would write over the gold set, or ask a provider for zero conversations, or read
// a tokenizer that does not exist, should be a 400 on submit rather than a failed
// row after the first call comes back.
async fn prepare_synth(pool: &PgPool, config: Value) -> Result<(String, Value), HttpResponse> {
    let parsed: crate::synth::SynthRequest = match serde_json::from_value(config) {
        Ok(config) => config,
        Err(e) => {
            return Err(HttpResponse::BadRequest().body(format!("invalid generation config: {e}")));
        }
    };

    if !parsed.base_url.starts_with("http://") && !parsed.base_url.starts_with("https://") {
        return Err(HttpResponse::BadRequest()
            .body("the API base must be an http(s) URL, e.g. https://openrouter.ai/api/v1"));
    }
    if parsed.model.trim().is_empty() {
        return Err(HttpResponse::BadRequest().body("name the model to generate with"));
    }
    if parsed.count < 1 {
        return Err(HttpResponse::BadRequest().body("count must be at least 1"));
    }
    if parsed.batch < 1 {
        return Err(HttpResponse::BadRequest().body("batch must be at least 1"));
    }
    if !(parsed.temperature.is_finite() && parsed.temperature > 0.0) {
        // Temperature 0 is argmax, and this is a diversity task: every call would
        // return the same conversations, all but the first deduplicated away.
        return Err(HttpResponse::BadRequest().body("temperature must be a positive number"));
    }

    // File *names*, never paths: a run is not a way to write anywhere on the disk
    // from a browser form.
    for name in [&parsed.out_file, &parsed.exemplar_file] {
        if name.contains('/') || name.contains('\\') || !name.ends_with(".jsonl") {
            return Err(HttpResponse::BadRequest()
                .body("file names must be plain .jsonl names inside the SFT directory"));
        }
    }
    // The gold set is hand-curated and irreplaceable; generated output goes beside
    // it, never into it.
    if parsed.out_file == parsed.exemplar_file {
        return Err(HttpResponse::BadRequest()
            .body("generated conversations cannot be written into the gold exemplar file"));
    }
    if !storage::sft_dir().join(&parsed.exemplar_file).exists() {
        return Err(HttpResponse::BadRequest().body(
            "no gold conversations to imitate — generation copies the style of real \
             ones, and with none to show it produces generic chat data",
        ));
    }

    // The model whose window the conversations have to fit. Optional, but checked
    // when present: a run configured against a deleted model would otherwise fail
    // on its first accepted conversation, minutes in.
    let mut label = format!("{} × {}", parsed.model, parsed.count);
    if let Some(id_model) = parsed.id_model {
        match db::model::find_by_id(pool, id_model).await {
            Ok(Some(model)) => label = format!("{} → {}", parsed.model, model.name),
            Ok(None) => return Err(HttpResponse::NotFound().body("model not found")),
            Err(e) => {
                eprintln!("synth model lookup db error: {}", e);
                return Err(HttpResponse::InternalServerError().body("failed to queue job"));
            }
        }
        let model_dir = storage::models_dir().join(id_model.to_string());
        if !model_dir.join("merges.bin").exists() {
            return Err(HttpResponse::BadRequest().body(
                "that model has no tokenizer on disk, so there is no way to measure \
                 whether a conversation fits its context window",
            ));
        }
    }

    match serde_json::to_value(&parsed) {
        Ok(value) => Ok((label, value)),
        Err(e) => {
            eprintln!("serialize synth config error: {}", e);
            Err(HttpResponse::InternalServerError().body("failed to queue job"))
        }
    }
}

// Where to move a queued job: directly after `after`, or to the front of the
// queued band when it is absent. Expressed as a neighbour rather than an index so
// a drop lands where it was dropped even if the board shifted underneath it.
#[derive(Deserialize)]
struct ReorderPayload {
    #[serde(default)]
    after: Option<Uuid>,
}

#[patch("/{id}")]
async fn reorder_job(
    _admin: AdminUser,
    path: web::Path<Uuid>,
    payload: web::Json<ReorderPayload>,
    pool: web::Data<PgPool>,
    queue: web::Data<Arc<Queue>>,
) -> impl Responder {
    let id_job = path.into_inner();

    match db::job::find_by_id(pool.get_ref(), id_job).await {
        Ok(Some(job)) if job.status == JobStatus::Queued => {}
        // A running job is at the front by definition, and a finished one has no
        // position to change.
        Ok(Some(_)) => return HttpResponse::Conflict().body("only a queued job can be reordered"),
        Ok(None) => return HttpResponse::NotFound().body("job not found"),
        Err(e) => {
            eprintln!("reorder job lookup db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to reorder job");
        }
    }

    match db::job::reorder(pool.get_ref(), id_job, payload.after).await {
        Ok(()) => {
            queue.notify_changed();
            HttpResponse::NoContent().finish()
        }
        Err(e) => {
            eprintln!("reorder job db error: {}", e);
            HttpResponse::InternalServerError().body("failed to reorder job")
        }
    }
}

// Cancels a job. A queued job stops existing as work immediately; a running one
// enters `cancelling` and stops at its worker's next step boundary, which is why
// this returns 202 rather than pretending the job is already over.
#[post("/{id}/cancel")]
async fn cancel_job(
    _admin: AdminUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
    queue: web::Data<Arc<Queue>>,
) -> impl Responder {
    let id_job = path.into_inner();

    // Tried first: if this succeeds the job never started, and there is no worker
    // to tell. The UPDATE is conditional on the row still being queued, so a job
    // the runner claimed a moment ago falls through to the running path instead of
    // being marked cancelled while its worker carries on.
    match db::job::cancel_queued(pool.get_ref(), id_job).await {
        Ok(true) => {
            queue.notify_changed();
            return HttpResponse::NoContent().finish();
        }
        Ok(false) => {}
        Err(e) => {
            eprintln!("cancel queued job db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to cancel job");
        }
    }

    if !queue.request_cancel(id_job) {
        // Not queued and not the job on the GPU: either already finished or never
        // existed. Distinguish the two so the UI can say which.
        return match db::job::find_by_id(pool.get_ref(), id_job).await {
            Ok(Some(_)) => HttpResponse::Conflict().body("this job has already finished"),
            Ok(None) => HttpResponse::NotFound().body("job not found"),
            Err(e) => {
                eprintln!("cancel job lookup db error: {}", e);
                HttpResponse::InternalServerError().body("failed to cancel job")
            }
        };
    }

    match db::job::request_cancel(pool.get_ref(), id_job).await {
        Ok(_) => {
            queue.notify_changed();
            HttpResponse::Accepted().body("stopping at the next step boundary")
        }
        Err(e) => {
            eprintln!("request cancel db error: {}", e);
            HttpResponse::InternalServerError().body("failed to cancel job")
        }
    }
}

// Queues a fresh copy of a finished job. The config is stored precisely so this
// works: most training runs are a tweak of the previous one, and re-running an
// identical config is the common case that shouldn't require retyping it.
#[post("/{id}/retry")]
async fn retry_job(
    admin: AdminUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
    queue: web::Data<Arc<Queue>>,
) -> impl Responder {
    let original = match db::job::find_by_id(pool.get_ref(), path.into_inner()).await {
        Ok(Some(job)) => job,
        Ok(None) => return HttpResponse::NotFound().body("job not found"),
        Err(e) => {
            eprintln!("retry job lookup db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to retry job");
        }
    };
    if !original.status.is_terminal() {
        return HttpResponse::Conflict().body("this job hasn't finished yet");
    }
    // A chat turn's reply already exists (or the turn was abandoned); re-running it
    // would append a second reply to the same question.
    if original.kind == JobKind::Chat {
        return HttpResponse::BadRequest().body("a chat turn cannot be re-run");
    }

    match db::job::create(
        pool.get_ref(),
        original.kind,
        &original.label,
        &original.config,
        Some(admin.claims.sub),
    )
    .await
    {
        Ok(job) => {
            queue.wake();
            queue.notify_changed();
            HttpResponse::Created().json(JobResponse::from(job))
        }
        Err(e) => {
            eprintln!("retry job db error: {}", e);
            HttpResponse::InternalServerError().body("failed to retry job")
        }
    }
}

// Removes a job from the board. Queued and finished rows go; a running one has to
// be cancelled first, because deleting the row its worker is about to write to
// would silently drop the outcome.
#[delete("/{id}")]
async fn delete_job(
    _admin: AdminUser,
    path: web::Path<Uuid>,
    pool: web::Data<PgPool>,
    queue: web::Data<Arc<Queue>>,
) -> impl Responder {
    let id_job = path.into_inner();

    match db::job::find_by_id(pool.get_ref(), id_job).await {
        Ok(Some(job)) if job.status == JobStatus::Running || job.status == JobStatus::Cancelling => {
            return HttpResponse::Conflict().body("cancel this job before deleting it");
        }
        Ok(Some(_)) => {}
        Ok(None) => return HttpResponse::NotFound().body("job not found"),
        Err(e) => {
            eprintln!("delete job lookup db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to delete job");
        }
    }

    match db::job::delete(pool.get_ref(), id_job).await {
        Ok(true) => {
            queue.notify_changed();
            HttpResponse::NoContent().finish()
        }
        Ok(false) => HttpResponse::NotFound().body("job not found"),
        Err(e) => {
            eprintln!("delete job db error: {}", e);
            HttpResponse::InternalServerError().body("failed to delete job")
        }
    }
}

#[derive(Deserialize)]
struct PausePayload {
    paused: bool,
}

// Stops or resumes the runner. Pausing never interrupts the job already on the
// GPU — it only stops the next one being claimed, which is what makes it safe to
// hit mid-run when you want the device back after the current job.
#[post("/pause")]
async fn set_pause(
    _admin: AdminUser,
    payload: web::Json<PausePayload>,
    pool: web::Data<PgPool>,
    queue: web::Data<Arc<Queue>>,
) -> impl Responder {
    match db::job::set_paused(pool.get_ref(), payload.paused).await {
        Ok(()) => {
            // Resuming has to wake the runner, which is otherwise parked.
            queue.wake();
            queue.notify_changed();
            HttpResponse::NoContent().finish()
        }
        Err(e) => {
            eprintln!("set pause db error: {}", e);
            HttpResponse::InternalServerError().body("failed to update the queue")
        }
    }
}

// ── the socket ──────────────────────────────────────────────────────────────

// Live updates. Same handshake as the chat socket — auth in the first frame — then
// a subscribe frame choosing between the whole board and one job's frames.
#[get("/ws")]
async fn jobs_ws(
    req: HttpRequest,
    body: web::Payload,
    queue: web::Data<Arc<Queue>>,
    jwt_secret: web::Data<String>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, session, msg_stream) = actix_ws::handle(&req, body)?;
    let queue = queue.get_ref().clone();
    let secret = jwt_secret.get_ref().clone();
    actix_web::rt::spawn(jobs_session(session, msg_stream, queue, secret));
    Ok(response)
}

// `{ "subscribe": "queue" }` follows the board; `{ "job_id": "<uuid>" }` follows
// one job's frames.
#[derive(Deserialize)]
struct SubscribeFrame {
    #[serde(default)]
    subscribe: Option<String>,
    #[serde(default)]
    job_id: Option<Uuid>,
}

#[derive(Deserialize)]
struct ActionFrame {
    action: String,
}

async fn jobs_session(
    mut session: actix_ws::Session,
    mut stream: actix_ws::MessageStream,
    queue: Arc<Queue>,
    secret: String,
) {
    use actix_ws::Message;

    // ---- authenticate on the first text frame ----
    loop {
        match stream.next().await {
            Some(Ok(Message::Text(text))) => match ws::authenticate(&text, &secret) {
                Some(_claims) => break,
                None => {
                    // Best-effort notify + close; we return regardless of whether
                    // the peer is still there to receive them.
                    let _ = session.text(ws::error_frame("authentication failed")).await;
                    let _ = session.close(None).await;
                    return;
                }
            },
            Some(Ok(Message::Ping(bytes))) => {
                // Best-effort pong; a dead peer ends the loop on the next read.
                let _ = session.pong(&bytes).await;
            }
            Some(Ok(Message::Close(_))) | None => return,
            _ => {}
        }
    }
    let _ = session
        .text(serde_json::json!({ "type": "ready" }).to_string())
        .await;

    // ---- subscribe frame ----
    let subscription = loop {
        match stream.next().await {
            Some(Ok(Message::Text(text))) => {
                match serde_json::from_str::<SubscribeFrame>(&text) {
                    Ok(frame) if frame.subscribe.as_deref() == Some("queue") => {
                        break Subscription::Board;
                    }
                    Ok(frame) => match frame.job_id {
                        Some(id) => break Subscription::One(id),
                        None => {
                            let _ = session.text(ws::error_frame("invalid subscribe frame")).await;
                            let _ = session.close(None).await;
                            return;
                        }
                    },
                    Err(_) => {
                        let _ = session.text(ws::error_frame("invalid subscribe frame")).await;
                        let _ = session.close(None).await;
                        return;
                    }
                }
            }
            Some(Ok(Message::Ping(bytes))) => {
                let _ = session.pong(&bytes).await;
            }
            Some(Ok(Message::Close(_))) | None => return,
            _ => {}
        }
    };

    match subscription {
        Subscription::Board => board_session(session, stream, queue).await,
        Subscription::One(id_job) => job_session(session, stream, queue, id_job).await,
    }
}

enum Subscription {
    Board,
    One(Uuid),
}

// Forwards board-level change events. The events are nudges rather than payloads —
// the client refetches `GET /jobs` — so there is exactly one description of a job
// and no second shape to keep in step with it.
async fn board_session(
    mut session: actix_ws::Session,
    mut stream: actix_ws::MessageStream,
    queue: Arc<Queue>,
) {
    use actix_ws::Message;
    let mut rx = queue.subscribe();

    loop {
        tokio::select! {
            incoming = stream.next() => match incoming {
                Some(Ok(Message::Ping(bytes))) => {
                    let _ = session.pong(&bytes).await;
                }
                Some(Ok(Message::Close(_))) | None => break,
                _ => {}
            },
            event = rx.recv() => match event {
                Ok(text) => {
                    if session.text(text).await.is_err() {
                        break;
                    }
                }
                // A board that fell behind has missed only nudges; one more tells
                // the client to refetch, which resolves any amount of lag.
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    let nudge = serde_json::json!({ "type": "changed" }).to_string();
                    if session.text(nudge).await.is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            },
        }
    }

    let _ = session.close(None).await;
}

// Follows one job: a catch-up snapshot, then its live frames until it ends.
// Closing this socket does NOT cancel the job — only `{"action":"cancel"}` does.
async fn job_session(
    mut session: actix_ws::Session,
    mut stream: actix_ws::MessageStream,
    queue: Arc<Queue>,
    id_job: Uuid,
) {
    use actix_ws::Message;

    // Subscribe while the live handles are in hand so a frame can't slip between
    // reading the snapshot and subscribing — a duplicate frame is harmless (the
    // client renders idempotently), a missed one is not.
    let attached = queue.live_for(id_job).map(|live| {
        let rx = live.tx.subscribe();
        (live.snapshot_frame(), rx)
    });

    let (snapshot, mut rx) = match attached {
        Some(attached) => attached,
        // Not the job on the GPU: it is queued, or already finished. Either way
        // there are no frames to follow, and the REST row is the whole story.
        None => {
            let _ = session
                .text(serde_json::json!({ "type": "inactive" }).to_string())
                .await;
            let _ = session.close(None).await;
            return;
        }
    };

    if session.text(snapshot).await.is_err() {
        return;
    }

    loop {
        tokio::select! {
            biased;
            incoming = stream.next() => match incoming {
                Some(Ok(Message::Text(text))) => {
                    if serde_json::from_str::<ActionFrame>(&text)
                        .map(|f| f.action == "cancel")
                        .unwrap_or(false)
                    {
                        queue.request_cancel(id_job);
                    }
                }
                Some(Ok(Message::Ping(bytes))) => {
                    let _ = session.pong(&bytes).await;
                }
                Some(Ok(Message::Close(_))) | None => break,
                _ => {}
            },
            msg = rx.recv() => match msg {
                Ok(text) => {
                    // The worker's last frame is "done" or "error"; forward it,
                    // then close.
                    let terminal = text.contains("\"type\":\"done\"")
                        || text.contains("\"type\":\"error\"");
                    if session.text(text).await.is_err() {
                        break;
                    }
                    if terminal {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    // A slow socket fell behind; re-sync from the snapshot.
                    match queue.live_for(id_job) {
                        Some(live) => {
                            if session.text(live.snapshot_frame()).await.is_err() {
                                break;
                            }
                        }
                        None => break,
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            },
        }
    }

    let _ = session.close(None).await;
}

// ── validation ──────────────────────────────────────────────────────────────

// Rejects obviously-bad configs before a job is queued. model_name becomes a
// display label and a unique key, so it must stay legible.
fn validate_config(cfg: &TrainConfig) -> Result<(), &'static str> {
    let name = cfg.model_name.trim();
    if name.is_empty() {
        return Err("model_name is required");
    }
    if name.len() != cfg.model_name.len() {
        return Err("model_name may not have leading or trailing whitespace");
    }
    // The name is a display label + unique key only — weights live under the
    // model's UUID, not its name — so spaces and dots (e.g. "Opal 1.1") are fine.
    // Just reject control characters.
    if name.chars().any(|c| c.is_control()) {
        return Err("model_name may not contain control characters");
    }
    if cfg.dimensions <= 0
        || cfg.ctx_window <= 0
        || cfg.heads <= 0
        || cfg.layers <= 0
        || cfg.epochs <= 0
        || cfg.ff_multiplier <= 0
    {
        return Err("model parameters must be positive");
    }
    // dimensions and ctx_window are powers of two by design (GPU tiling); a
    // non-power-of-two slips through the kernels' block math.
    if !is_power_of_two(cfg.dimensions) {
        return Err("dimensions must be a power of two");
    }
    if !is_power_of_two(cfg.ctx_window) {
        return Err("ctx_window must be a power of two");
    }
    // head_dim = dimensions / heads must divide evenly — otherwise the per-head
    // slices don't tile back into `dimensions` and the attention kernels read
    // past their buffers.
    if cfg.dimensions % cfg.heads != 0 {
        return Err("dimensions must be divisible by heads");
    }
    if !(cfg.lr.is_finite() && cfg.lr > 0.0) {
        return Err("lr must be a positive number");
    }
    if cfg.max_vocab < 256 {
        return Err("max_vocab must be at least 256");
    }
    if cfg.mode == TrainMode::Kto && !(cfg.beta.is_finite() && cfg.beta > 0.0) {
        return Err("beta must be a positive number");
    }
    // Pretrain builds its corpus from the selected upload datasets; the two
    // fine-tuning modes ignore them (SFT trains on chat data, KTO on rated
    // replies), so only require a selection for pretrain. A curriculum run supplies
    // its datasets per stage instead of the flat list.
    if cfg.mode == TrainMode::Pretrain {
        if cfg.curriculum.is_empty() {
            if cfg.datasets.is_empty() {
                return Err("select at least one dataset to train on");
            }
            if cfg.datasets.iter().any(|d| d.name.trim().is_empty()) {
                return Err("dataset name is required");
            }
        } else {
            for stage in &cfg.curriculum {
                if stage.epochs <= 0 {
                    return Err("each curriculum stage needs at least one epoch");
                }
                if stage.datasets.is_empty() {
                    return Err("each curriculum stage needs at least one dataset");
                }
                if stage.datasets.iter().any(|d| d.name.trim().is_empty()) {
                    return Err("dataset name is required");
                }
            }
        }
    }
    Ok(())
}

fn is_power_of_two(n: i64) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

// Extra validation for the two fine-tuning runs (a no-op for pretrain): the base
// model to fine-tune from must be named and must resolve to an on-disk model (a
// `model.bin` under its UUID dir). A bad or missing base is a client error (400),
// not a server fault — catch it here so a job is never queued against a nonexistent
// base. Note: the model hyperparameters (dimensions/ctx_window/…) are ignored for a
// fine-tune — they are loaded from the base — but must still pass
// `validate_config`; the frontend mirrors the base's values into the request
// (ISSUE-0038).
fn validate_sft(cfg: &TrainConfig) -> Result<(), &'static str> {
    if cfg.mode == TrainMode::Pretrain {
        return Ok(());
    }
    let base_id = cfg
        .base_model_id
        .ok_or("base_model_id is required for a fine-tuning run")?;
    let base_dir = storage::models_dir().join(base_id.to_string());
    if !base_dir.join("model.bin").is_file() {
        return Err("base_model_id does not resolve to an existing model");
    }
    Ok(())
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    // jobs_ws (GET /jobs/ws) and the pause route must be registered before get_job
    // (GET /jobs/{id}), or "ws" and "pause" would be captured by the "{id}" pattern.
    cfg.service(jobs_ws);
    cfg.service(set_pause);
    cfg.service(list_jobs);
    cfg.service(create_job);
    cfg.service(get_job);
    cfg.service(reorder_job);
    cfg.service(cancel_job);
    cfg.service(retry_job);
    cfg.service(delete_job);
}

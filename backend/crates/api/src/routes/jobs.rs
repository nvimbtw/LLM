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
use jobs::queue::Queue;
use storage;
use training::{TrainConfig, TrainMode};
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
    // What the run produced, for a kind that has no table of its own to read back
    // from. Null for every other kind — see `db::job::set_result`.
    result: Option<Value>,
    // The job this one was queued behind, for a `waiting` row. Surfaced so the board
    // can draw a chain as a chain rather than as unrelated rows that mysteriously
    // never start.
    depends_on: Option<Uuid>,
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
            result: job.result,
            depends_on: job.depends_on,
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
//
// `then` is the same structure again, one step down the chain: jobs to run *after*
// this one, against the model it produces. They are validated now and filed now —
// the whole point is that nobody is awake later to file them — but they are filed
// `waiting`, without a model, because the model does not exist yet. `jobs::chain`
// fills it in when this job lands.
#[derive(Deserialize)]
struct CreatePayload {
    kind: JobKind,
    config: Value,
    // Recursive through the `Vec`, which is already an indirection, so no `Box`.
    #[serde(default)]
    then: Vec<CreatePayload>,
}

// How deep a chain may go. Not a technical limit — the resolver is iterative — but
// a bound on what one request can file, and eight overnight stages is already more
// than the GPU will get through.
const MAX_CHAIN_DEPTH: usize = 8;

// One job the request asked for, validated and ready to insert. `parent` indexes
// earlier in the same list, which is what makes the whole chain insertable in one
// pass: a node is always prepared, and inserted, after the job it follows.
struct PreparedJob {
    kind: JobKind,
    label: String,
    config: Value,
    parent: Option<usize>,
}

#[post("")]
async fn create_job(
    admin: AdminUser,
    payload: web::Json<CreatePayload>,
    pool: web::Data<PgPool>,
    queue: web::Data<Arc<Queue>>,
) -> impl Responder {
    let CreatePayload { kind, config, then } = payload.into_inner();

    // A list, because one submit can queue more than one job: the benchmark's split
    // switch queues the whole budget sweep at once, so the three runs exist together
    // or not at all. Every other kind yields exactly one.
    let prepared: Vec<(String, Value)> = match kind {
        JobKind::Train => match prepare_train(pool.get_ref(), config, false).await {
            Ok(prepared) => vec![prepared],
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
            Ok(prepared) => vec![prepared],
            Err(response) => return response,
        },
        JobKind::Gather => match prepare_gather(pool.get_ref(), config).await {
            Ok(prepared) => vec![prepared],
            Err(response) => return response,
        },
        JobKind::Synth => match prepare_synth(pool.get_ref(), config).await {
            Ok(prepared) => vec![prepared],
            Err(response) => return response,
        },
        JobKind::Teach => match prepare_teach(pool.get_ref(), config).await {
            Ok(prepared) => vec![prepared],
            Err(response) => return response,
        },
        JobKind::Chat => {
            return HttpResponse::BadRequest().body("chat jobs are queued from the chat socket");
        }
    };

    let mut plan: Vec<PreparedJob> = prepared
        .into_iter()
        .map(|(label, config)| PreparedJob {
            kind,
            label,
            config,
            parent: None,
        })
        .collect();

    if !then.is_empty() {
        // A sweep is three jobs and a follow-up follows exactly one, so there is no
        // answer to which of the three it would run behind. Refused rather than
        // guessed at.
        if plan.len() != 1 {
            return HttpResponse::BadRequest()
                .body("a benchmark sweep cannot carry follow-up jobs: queue them against one run");
        }
        if let Err(response) = plan_followups(pool.get_ref(), &mut plan, then).await {
            return response;
        }
    }

    // Inserted in order, so a sweep queues low → med → high and runs in that order,
    // and a chain's parent always exists before the row that references it.
    //
    // The response carries the **first** job: the caller navigates to a job page, and
    // one of them has to be it. A client that queued a sweep or a chain knows to look
    // at the board for the rest — `id_series` ties a sweep together, `depends_on`
    // ties a chain.
    let mut created: Vec<db::job::Job> = Vec::with_capacity(plan.len());
    for node in &plan {
        let result = match node.parent {
            None => {
                db::job::create(
                    pool.get_ref(),
                    node.kind,
                    &node.label,
                    &node.config,
                    Some(admin.claims.sub),
                )
                .await
            }
            Some(parent) => {
                db::job::create_waiting(
                    pool.get_ref(),
                    node.kind,
                    &node.label,
                    &node.config,
                    Some(admin.claims.sub),
                    created[parent].id_job,
                )
                .await
            }
        };
        match result {
            Ok(job) => created.push(job),
            Err(e) => {
                eprintln!("create job db error: {}", e);
                // A partly-queued sweep or chain is still queued: the jobs already
                // created are real work the user asked for, and cancelling them from
                // here would race the runner, which may have started one. Report the
                // failure and leave what exists — the board shows exactly what was
                // queued.
                return HttpResponse::InternalServerError().body("failed to queue job");
            }
        }
    }

    queue.wake();
    queue.notify_changed();
    match created.into_iter().next() {
        Some(job) => HttpResponse::Created().json(JobResponse::from(job)),
        None => HttpResponse::InternalServerError().body("failed to queue job"),
    }
}

// Validates the follow-up tree and appends it to `plan`, parents before children.
//
// Breadth-first over an explicit worklist rather than recursion: an `async fn` that
// calls itself needs boxing, and the flat plan with parent indices is what the
// insert loop above wants anyway.
async fn plan_followups(
    pool: &PgPool,
    plan: &mut Vec<PreparedJob>,
    then: Vec<CreatePayload>,
) -> Result<(), HttpResponse> {
    // (index of the job this node follows, the node, how deep it sits).
    let mut pending: Vec<(usize, CreatePayload, usize)> =
        then.into_iter().map(|node| (0usize, node, 1usize)).collect();

    while let Some((parent, node, depth)) = pending.pop() {
        if depth > MAX_CHAIN_DEPTH {
            return Err(HttpResponse::BadRequest()
                .body(format!("a chain may be at most {MAX_CHAIN_DEPTH} jobs deep")));
        }

        let CreatePayload {
            kind,
            config,
            then: grandchildren,
        } = node;

        let parent_kind = plan[parent].kind;
        let parent_label = plan[parent].label.clone();
        let parent_storage = storage_id_of(&plan[parent]);

        let (label, config) = match kind {
            JobKind::Train => {
                let prepared = prepare_train(pool, config, true).await?;
                // A fine-tune that writes to its base's directory would overwrite,
                // mid-chain, the weights it is training from. The names differ by one
                // character often enough ("Balanced 2" → "Balanced 2.01") that this is
                // a real mistake to make at 1am, and it is unrecoverable.
                if let (Some(parent_storage), Some(child_storage)) =
                    (parent_storage, storage_id_in(&prepared.1))
                {
                    if parent_storage == child_storage {
                        return Err(HttpResponse::BadRequest().body(
                            "a follow-up run must have its own model name: this one would overwrite the model it fine-tunes from",
                        ));
                    }
                }
                prepared
            }
            JobKind::Benchmark => prepare_benchmark_future(config, &parent_label)?,
            // Only these two can inherit a model from the job they follow (see
            // `jobs::chain::inject_model`). The others name their model in a field
            // this would have to learn, and none of them is something to run
            // unattended behind a training run.
            _ => {
                return Err(HttpResponse::BadRequest()
                    .body("only training runs and benchmarks can be queued as follow-ups"));
            }
        };

        // Everything except a pretrain consumes a model from the job before it. Two
        // kinds can supply one: a training run produces a model, and a benchmark
        // passes on the model it scored — which is what makes "fine-tune it, score
        // language, score assistant" a straight line rather than a tree. See
        // `jobs::chain::model_of`.
        if needs_parent_model(kind, &config)
            && !matches!(parent_kind, JobKind::Train | JobKind::Benchmark)
        {
            return Err(HttpResponse::BadRequest().body(
                "a follow-up that needs a model must follow a training run or a benchmark",
            ));
        }

        plan.push(PreparedJob {
            kind,
            label,
            config,
            parent: Some(parent),
        });
        let index = plan.len() - 1;
        for grandchild in grandchildren {
            pending.push((index, grandchild, depth + 1));
        }
    }
    Ok(())
}

// Whether a follow-up will need the model its parent produces. Mirrors
// `jobs::chain::needs_model`, which is what actually acts on it at promotion; this
// copy exists so the impossible chain is a 400 at submit rather than a cancelled row
// hours later.
fn needs_parent_model(kind: JobKind, config: &Value) -> bool {
    match kind {
        // A pretrain starts from random weights and builds its own tokenizer, so
        // chaining one is pure sequencing: "run this after that, on the GPU I am not
        // awake to share out."
        JobKind::Train => config
            .get("mode")
            .and_then(|v| v.as_str())
            .is_some_and(|mode| mode != "pretrain"),
        _ => true,
    }
}

// Which model directory this job will hand to the step after it, as far as can be
// known at submit — enough for the overwrite check below.
//
// A *follow-up* benchmark has no `id_model` yet (it is injected at promotion), so
// this reads None for one and the check is skipped rather than guessed at. It still
// catches the case that matters: a fine-tune, a benchmark of it, then a second
// fine-tune reusing the first one's name.
fn storage_id_of(job: &PreparedJob) -> Option<Uuid> {
    match job.kind {
        JobKind::Train => storage_id_in(&job.config),
        JobKind::Benchmark => job
            .config
            .get("id_model")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok()),
        _ => None,
    }
}

fn storage_id_in(config: &Value) -> Option<Uuid> {
    config
        .get("storage_id")
        .and_then(|v| v.as_str())
        .and_then(|s| Uuid::parse_str(s).ok())
}

// Validates a benchmark queued as somebody's follow-up.
//
// Separate from `prepare_benchmark` rather than a flag on it, because almost none of
// that function applies: there is no model to look up, no tokenizer to check on disk
// and no sweep to expand. What is left is the suite, the budget, and a label built
// from the run this one follows — which names the model, since a training job's
// label is its model name.
fn prepare_benchmark_future(
    config: Value,
    parent_label: &str,
) -> Result<(String, Value), HttpResponse> {
    let parsed: BenchmarkConfig = match serde_json::from_value(config) {
        Ok(config) => config,
        Err(e) => {
            return Err(HttpResponse::BadRequest().body(format!("invalid benchmark config: {e}")));
        }
    };

    // The model is the whole reason this is a follow-up. Naming one would make it an
    // ordinary benchmark filed in a confusing place.
    if !parsed.id_model.is_nil() {
        return Err(HttpResponse::BadRequest().body(
            "a follow-up benchmark scores the model the job before it produces; it cannot name its own",
        ));
    }
    // Three runs cannot come out of one waiting row, and splitting a chain into
    // three at promotion would be a second kind of fan-out for one feature.
    if parsed.split_thinking {
        return Err(HttpResponse::BadRequest()
            .body("a thinking-budget sweep cannot be queued as a follow-up: queue it afterwards"));
    }
    let splittable = parsed.suite != db::benchmark::BenchmarkSuite::Language;
    if parsed.budget.is_some() && !splittable {
        return Err(HttpResponse::BadRequest().body(
            "the language suite has no thinking budget to set: none of its stages render a chat prompt",
        ));
    }

    let label = match parsed.budget {
        Some(b) => format!("{} · {} · {}", parent_label, parsed.suite.as_str(), b.as_str()),
        None => format!("{} · {}", parent_label, parsed.suite.as_str()),
    };
    Ok((
        label,
        serde_json::json!({
            "suite": parsed.suite,
            "budget": parsed.budget,
            "id_series": Option::<Uuid>::None,
        }),
    ))
}

// Validates a training config and resolves the storage id the run will write to,
// returning the label and the config as it will be stored.
//
// `inherits_base` marks a run queued as somebody's follow-up: its base model is the
// one the job before it will produce, so it has no `base_model_id` yet and the two
// checks that read the base off disk cannot run. Everything else — the
// hyperparameters, the KTO/RAFT data gates, the storage id — is as true now as it
// will be then, and is checked now.
async fn prepare_train(
    pool: &PgPool,
    config: Value,
    inherits_base: bool,
) -> Result<(String, Value), HttpResponse> {
    let mut config: TrainConfig = match serde_json::from_value(config) {
        Ok(config) => config,
        Err(e) => return Err(HttpResponse::BadRequest().body(format!("invalid training config: {}", e))),
    };
    if let Err(msg) = validate_config(&config) {
        return Err(HttpResponse::BadRequest().body(msg));
    }
    if inherits_base {
        // The base is whatever the job before this one produces. Naming one as well
        // would be two answers to the same question, and the chain's answer is the
        // one that wins at promotion — so the request is refused rather than having
        // its explicit base silently overwritten.
        if config.base_model_id.is_some() {
            return Err(HttpResponse::BadRequest().body(
                "a follow-up fine-tune trains on the model the job before it produces; it cannot name its own base",
            ));
        }
    } else if let Err(msg) = validate_sft(&config) {
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
    // Pretrain always strips (see `training::strip_thinking_for`), so record that
    // rather than whatever the form sent. The stored config is what
    // `training::model_reasoning` later reads to decide how to render this model's
    // held-out replies at benchmark time — a pretrain row claiming `false` would
    // have its assistant suite scored against thinking blocks its corpus never had.
    if config.mode == TrainMode::Pretrain {
        config.strip_thinking = true;
    }

    // Skipped for a follow-up: there is no base on disk to read yet.
    // `jobs::chain::inject_model` does exactly this at promotion instead, once there
    // is, so the queued row is true from the moment it can be.
    if config.mode != TrainMode::Pretrain && !inherits_base {
        if let Some(base_id) = config.base_model_id {
            let base_dir = storage::models_dir().join(base_id.to_string());
            if let Err(e) = training::apply_base_architecture(&mut config, &base_dir) {
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
        if let Err(msg) = labeling::gather::check_signals(counts.desirable, counts.undesirable) {
            return Err(HttpResponse::BadRequest().body(msg));
        }

        let (auto_up, auto_down) =
            labeling::autolabel::catalogue_counts(config.catalogue_scope());
        let (approved, rejected) =
            (counts.desirable + auto_up, counts.undesirable + auto_down);
        if let Err(msg) = labeling::gather::check_catalogue(approved, rejected) {
            return Err(HttpResponse::BadRequest().body(msg));
        }

        // The backstop under the catalogue bar. Clamped rather than refused: the
        // config is otherwise fine, and silently running it five times over a
        // small catalogue is the failure — not the request itself.
        let cap = labeling::gather::max_epochs_for(approved + rejected);
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
        let (auto_up, _) = labeling::autolabel::catalogue_counts(config.catalogue_scope());
        let approved = counts.desirable + auto_up;
        if let Err(msg) = labeling::gather::check_raft(approved) {
            return Err(HttpResponse::BadRequest().body(msg));
        }
        let cap = labeling::gather::max_epochs_for(approved);
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
    // Defaulted to the nil UUID rather than required, because a benchmark queued as
    // a *follow-up* legitimately has no model yet — it scores whatever the job
    // before it produces. Both readers check for nil, in opposite directions.
    #[serde(default)]
    id_model: Uuid,
    // Which of the two suites to run. Defaulted rather than required so a client that
    // predates the split still queues the language bench, which is what it meant.
    #[serde(default)]
    suite: db::benchmark::BenchmarkSuite,
    /// Render every held-out conversation under one forced thinking budget. Absent is
    /// the unbudgeted run — each conversation keeps the marker it was written with —
    /// which is what every config before this meant and still means.
    #[serde(default)]
    budget: Option<llm::chat_template::ThinkBudget>,
    /// Queue the whole sweep: one run per budget, sharing a series id so the chart
    /// draws them as one line. Ignored on the language suite, which renders no chat
    /// prompt and would produce three identical scores.
    #[serde(default)]
    split_thinking: bool,
}

// Checks the model exists and has a tokenizer now, so an impossible benchmark is
// refused at submit rather than occupying a queue slot to fail.
//
// Returns **one job, or three**. With `split_thinking` the run is swept across the
// three thinking budgets: three rows sharing an `id_series`, which is what lets the
// chart draw them as one connected line rather than three unrelated dots.
//
// Three jobs rather than one job producing three runs, because the queue's unit is
// already "one job, one run, one row": three jobs get three progress bars, three
// cancel buttons, and a failure in the middle of the sweep leaves the other two
// standing. Folding the sweep into one job would need a second notion of progress
// inside the runner and would make cancelling the high tier mean cancelling all of it.
async fn prepare_benchmark(
    pool: &PgPool,
    config: Value,
) -> Result<Vec<(String, Value)>, HttpResponse> {
    let parsed: BenchmarkConfig = match serde_json::from_value(config) {
        Ok(config) => config,
        Err(_) => return Err(HttpResponse::BadRequest().body("a benchmark job needs an id_model")),
    };
    if parsed.id_model.is_nil() {
        return Err(HttpResponse::BadRequest().body("a benchmark job needs an id_model"));
    }

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

    // The language suite renders no chat prompt in any of its four stages — corpus
    // perplexity and grammar are raw text, generation continues bare prompts, and
    // preference passes no budget — so a marker cannot reach it. Sweeping it would
    // queue three runs that produce the same score three times, which reads as a
    // broken feature rather than as an inapplicable one. Refused at submit, where it
    // can be explained, rather than silently collapsed to one run.
    let splittable = parsed.suite != db::benchmark::BenchmarkSuite::Language;
    if (parsed.split_thinking || parsed.budget.is_some()) && !splittable {
        return Err(HttpResponse::BadRequest().body(
            "the language suite has no thinking budget to set: none of its stages              render a chat prompt, so all three budgets would score identically",
        ));
    }

    // The suite is in the label because every suite shows up in one job history and
    // one board, and "Balanced 1" twice tells you nothing about which is which. A
    // swept run carries its budget for the same reason, three times over.
    let label = |budget: Option<llm::chat_template::ThinkBudget>| match budget {
        Some(b) => format!("{} · {} · {}", model.name, parsed.suite.as_str(), b.as_str()),
        None => format!("{} · {}", model.name, parsed.suite.as_str()),
    };
    let job = |budget: Option<llm::chat_template::ThinkBudget>, series: Option<Uuid>| {
        (
            label(budget),
            serde_json::json!({
                "id_model": parsed.id_model,
                "suite": parsed.suite,
                "budget": budget,
                "id_series": series,
            }),
        )
    };

    if !parsed.split_thinking {
        return Ok(vec![job(parsed.budget, None)]);
    }

    // One id for the sweep, minted here because this is the only place that knows
    // the three runs are one thing.
    let series = Uuid::new_v4();
    Ok(llm::chat_template::ThinkBudget::ALL
        .iter()
        .map(|budget| job(Some(*budget), Some(series)))
        .collect())
}

// Validates an auto-labelling config: the model that will generate the candidates
// must exist and have a tokenizer on disk, and the run must ask for enough
// candidates per prompt to have a best and a worst.
async fn prepare_autolabel(pool: &PgPool, config: Value) -> Result<(String, Value), HttpResponse> {
    let parsed: labeling::autolabel::AutolabelRequest = match serde_json::from_value(config) {
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
    let parsed: labeling::gather::GatherRequest = match serde_json::from_value(config) {
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
    let parsed: synth::SynthRequest = match serde_json::from_value(config) {
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

// Validates a local-teacher request.
//
// The `prepare_synth` twin, and it checks a different set of things because the
// failure modes are different: there is no provider or API key, but there *is* a
// weights file that has to exist on disk and a set of tier thresholds that have to
// be ordered.
async fn prepare_teach(pool: &PgPool, config: Value) -> Result<(String, Value), HttpResponse> {
    let parsed: synth::teach::TeachRequest = match serde_json::from_value(config) {
        Ok(config) => config,
        Err(e) => {
            return Err(HttpResponse::BadRequest().body(format!("invalid teaching config: {e}")));
        }
    };

    // A file name, never a path — the same rule as the SFT file names below, for
    // the same reason. `teacher::Teacher::load` enforces it again at load time; a
    // 400 here is so a person finds out on submit.
    let model_file = parsed.teacher.model_file.trim();
    if model_file.is_empty() {
        return Err(HttpResponse::BadRequest().body("name the teacher model to generate with"));
    }
    if model_file.contains('/') || model_file.contains('\\') || model_file.contains("..") {
        return Err(HttpResponse::BadRequest()
            .body("the teacher model must be a plain file name inside the teacher directory"));
    }
    let model_path = storage::teacher_dir().join(model_file);
    if !model_path.is_file() {
        return Err(HttpResponse::BadRequest().body(format!(
            "no teacher model named {model_file} — put a .gguf in {}",
            storage::teacher_dir().display()
        )));
    }

    if parsed.count < 1 {
        return Err(HttpResponse::BadRequest().body("count must be at least 1"));
    }
    if parsed.teacher.n_ctx < 1024 {
        // The prompt alone carries several exemplars; below this there is no room
        // for three answers and the run would fail on its first call.
        return Err(HttpResponse::BadRequest()
            .body("n_ctx must be at least 1024 to hold the prompt and three answers"));
    }
    if !(parsed.sample.temperature.is_finite() && parsed.sample.temperature > 0.0) {
        // Temperature 0 is argmax, and this is a diversity task: every call would
        // return the same triple, all but the first deduplicated away.
        return Err(HttpResponse::BadRequest().body("temperature must be a positive number"));
    }
    // Thresholds define three ranges, so they have to be strictly increasing. Equal
    // bounds would make the middle tier unreachable and silently produce a two-tier
    // corpus, which is exactly the failure this feature exists to avoid.
    //
    // `0/0` is not a pair of bounds, it is the request to derive them from the
    // student's context window — the default, and what the composer sends. It has
    // to be allowed through here or the only correct configuration is the one the
    // route rejects.
    let derive_bounds = parsed.low_max_tokens == 0 && parsed.med_max_tokens == 0;
    if !derive_bounds && parsed.low_max_tokens >= parsed.med_max_tokens {
        return Err(HttpResponse::BadRequest().body(
            "the low thinking budget must be smaller than the medium one, or the \
             middle tier can never be filled",
        ));
    }

    if parsed.exemplar_file.contains('/')
        || parsed.exemplar_file.contains('\\')
        || !parsed.exemplar_file.ends_with(".jsonl")
    {
        return Err(HttpResponse::BadRequest()
            .body("file names must be plain .jsonl names inside the SFT directory"));
    }
    if !storage::sft_dir().join(&parsed.exemplar_file).exists() {
        return Err(HttpResponse::BadRequest().body(
            "no gold conversations to imitate — generation copies the style of real \
             ones, and with none to show it produces generic chat data",
        ));
    }
    // The tier files are what a run writes; reading exemplars from one would feed a
    // run its own output and collapse the style onto itself.
    for tier in synth::teach::Tier::ALL {
        if parsed.exemplar_file == synth::teach::tier_file_name(tier) {
            return Err(HttpResponse::BadRequest().body(
                "that file is where this job writes its output — imitating it would \
                 feed the run its own generations",
            ));
        }
    }

    let mut label = format!("{} × {}", model_file, parsed.count);
    if let Some(id_model) = parsed.id_model {
        match db::model::find_by_id(pool, id_model).await {
            Ok(Some(model)) => label = format!("{} → {}", model_file, model.name),
            Ok(None) => return Err(HttpResponse::NotFound().body("model not found")),
            Err(e) => {
                eprintln!("teach model lookup db error: {}", e);
                return Err(HttpResponse::InternalServerError().body("failed to queue job"));
            }
        }
        let model_dir = storage::models_dir().join(id_model.to_string());
        if !model_dir.join("merges.bin").exists() {
            return Err(HttpResponse::BadRequest().body(
                "that model has no tokenizer on disk, so there is no way to measure a \
                 thinking budget in its tokens",
            ));
        }
    }

    match serde_json::to_value(&parsed) {
        Ok(value) => Ok((label, value)),
        Err(e) => {
            eprintln!("serialize teach config error: {}", e);
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
            // This path never reaches the runner, so the resolver has to be called
            // here: without it everything queued behind this job waits forever on a
            // model that will never be built.
            settle_chain_after(pool.get_ref(), queue.get_ref(), id_job).await;
            queue.notify_changed();
            return HttpResponse::NoContent().finish();
        }
        Ok(false) => {}
        Err(e) => {
            eprintln!("cancel queued job db error: {}", e);
            return HttpResponse::InternalServerError().body("failed to cancel job");
        }
    }

    // A waiting job is cancellable too — it is work the user asked for and can
    // change their mind about — and cancelling it takes everything chained behind it
    // with it.
    match db::job::cancel_waiting(pool.get_ref(), id_job, "cancelled before it started").await {
        Ok(true) => {
            settle_chain_after(pool.get_ref(), queue.get_ref(), id_job).await;
            queue.notify_changed();
            return HttpResponse::NoContent().finish();
        }
        Ok(false) => {}
        Err(e) => {
            eprintln!("cancel waiting job db error: {}", e);
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

// Settles whatever was queued behind a job that has just been closed out by a route
// rather than by the runner.
//
// The queue calls `jobs::chain::resolve` itself after every job it finishes, but the
// two routes below close a job out without it ever having run, and a follow-up left
// waiting on a job that will never produce anything is a row that sits on the board
// forever. Best-effort: the job is already cancelled or deleted correctly, and the
// boot reconcile is the backstop.
async fn settle_chain_after(pool: &PgPool, queue: &Arc<Queue>, id_job: Uuid) {
    match db::job::find_by_id(pool, id_job).await {
        Ok(Some(job)) => {
            jobs::chain::resolve(pool, queue, &job).await;
        }
        // Deleted: the FK has already nulled its children's `depends_on`, so they are
        // orphans rather than a chain with a head to consult.
        Ok(None) => match db::job::list_orphaned_waiting(pool).await {
            Ok(orphans) => {
                jobs::chain::cancel_orphans(pool, queue, &orphans).await;
            }
            Err(e) => eprintln!("orphaned follow-up lookup db error: {}", e),
        },
        Err(e) => eprintln!("chain settle lookup db error: {}", e),
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
            // `ON DELETE SET NULL` has just orphaned anything chained behind this
            // job. A follow-up whose base model will now never be built can only
            // fail later and more expensively, so it is cancelled here with that as
            // the recorded reason.
            settle_chain_after(pool.get_ref(), queue.get_ref(), id_job).await;
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

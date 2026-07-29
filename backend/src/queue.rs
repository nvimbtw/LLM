// The job queue: two runners, each taking one job at a time from its own lane, in
// `queue_rank` order.
//
// There is exactly one device, so there is exactly one line for it. Before this
// module that rule was enforced by route handlers asking two in-memory registries
// whether they were busy — true only for the paths that remembered to ask (chat
// never did), and forgotten entirely across a restart. Now the rule is structural:
// this loop is the only thing that starts GPU work, and it runs one job to
// completion before claiming the next.
//
// The runner is an async task that owns the pool, which is what lets every durable
// write happen here: claiming, coarse progress, the terminal row, and each kind's
// domain record (the model, the benchmark run, the assistant message). The workers
// themselves are blocking CUDA calls on plain OS threads and never touch the pool.
//
// Live state — the cancel flag, the frame broadcast, the accumulated snapshot —
// stays in memory, held in `live` for exactly as long as a job is running. Nothing
// in flight survives a restart, so there is nothing to persist about it; what
// survives is the queue itself.
//
// **Why there are two lines rather than one.** Every kind above is FFI into CUDA,
// so "one runner" and "one device" were the same statement. Synthetic data
// generation is not: it waits on somebody else's HTTP endpoint for minutes and uses
// no local compute at all. Sharing the device's line would mean a generation run
// idles the GPU and blocks chat for as long as it lasts, which is strictly worse
// than not having the feature. So a job declares a `JobLane`, each lane has its own
// runner, and the two contend for nothing — `live` therefore holds up to one job
// *per lane* rather than one job overall.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sqlx::PgPool;
use tokio::sync::{Notify, broadcast, mpsc, oneshot};
use uuid::Uuid;

use crate::autolabel;
use db;
use db::job::{JobKind, JobLane, JobStatus};
use crate::gather;
use storage;
use crate::synth;
use crate::train_worker::{self, TrainCtx, TrainPhase, TrainState};

const BROADCAST_CAPACITY: usize = 256;
// How often coarse progress reaches the durable row. The live frames go out at
// whatever rate the worker produces them; this is only so a fresh page load has a
// sane bar before its socket attaches.
const PROGRESS_WRITE_INTERVAL: Duration = Duration::from_secs(2);
// A fallback tick behind the wake signal. Everything that should start a job also
// calls `wake`, so this only matters if a notification is missed or a row appears
// from outside this process — cheap insurance against a queue that sleeps forever.
const IDLE_POLL: Duration = Duration::from_secs(5);

// Sampling knobs for a chat turn. A reply runs until the model emits
// <|endoftext|>; MAX_TOKENS is only a safety ceiling so a model that never emits
// it can't generate forever (the sliding context window means it otherwise
// wouldn't stop on its own).
const MAX_TOKENS: usize = 4096;
const TEMPERATURE: f32 = 0.8;
const TOP_K: u32 = 40;

// One coarse progress report from a worker thread.
pub struct ProgressTick {
    pub fraction: f64,
    pub stage: String,
}

// The accumulated state a socket needs when it attaches to a run already in
// flight. Training's is a rich record of series, diagnostics and log lines;
// benchmark and chat have a stage line and (for chat) the text so far, so they
// share the simpler shape.
pub enum Snapshot {
    Train(Arc<TrainCtx>),
    Simple(Arc<Mutex<SimpleState>>),
    // A generation run accumulates two series and a feed, and a page that attaches
    // ten minutes in should see the run so far rather than an empty chart — so its
    // snapshot carries the whole accumulation, like training's does.
    Synth(Arc<Mutex<SynthState>>),
}

// Everything a synthetic-data run has produced so far.
pub struct SynthState {
    pub stage: String,
    pub fraction: f64,
    pub calls: Vec<synth::CallReport>,
    // The most recent conversations, in full. A live window rather than a log: the
    // text is what makes the feed worth watching and also what makes it heavy.
    pub accepted: Vec<synth::Accepted>,
    pub added: usize,
    pub target: usize,
    pub duplicates: usize,
    pub with_thinking: usize,
}

// How many accepted conversations the snapshot keeps. A screenful and a bit.
const SYNTH_FEED: usize = 24;

pub struct SimpleState {
    pub stage: String,
    pub fraction: f64,
    // Text generated so far, for a chat job. Empty for a benchmark.
    pub text: String,
}

// Handles onto a job that is running right now. Dropped the moment it finishes, so
// an entry here is the same statement as "this job's lane is busy with it".
pub struct Live {
    pub id_job: Uuid,
    pub cancel: Arc<AtomicBool>,
    pub tx: broadcast::Sender<String>,
    pub snapshot: Snapshot,
}

impl Live {
    // The catch-up frame for a socket attaching mid-run.
    pub fn snapshot_frame(&self) -> String {
        match &self.snapshot {
            Snapshot::Train(ctx) => train_worker::snapshot_frame(&lock(&ctx.state)),
            Snapshot::Simple(state) => {
                let st = lock(state);
                serde_json::json!({
                    "type": "snapshot",
                    "status": "running",
                    "stage": st.stage,
                    "fraction": st.fraction,
                    "text": st.text,
                })
                .to_string()
            }
            Snapshot::Synth(state) => {
                let st = lock(state);
                serde_json::json!({
                    "type": "snapshot",
                    "status": "running",
                    "stage": st.stage,
                    "fraction": st.fraction,
                    "calls": st.calls,
                    "accepted": st.accepted,
                    "added": st.added,
                    "target": st.target,
                    "duplicates": st.duplicates,
                    "with_thinking": st.with_thinking,
                })
                .to_string()
            }
        }
    }
}

pub struct Queue {
    pool: PgPool,
    // Wakes a runner when a job is enqueued, cancelled, or the queue is resumed.
    // One per lane: `notify_one` parks a permit when nobody is waiting, and a single
    // shared `Notify` would let the wrong runner consume the other's wake-up — a
    // queued chat turn would then sit until the 5s fallback tick.
    wake_device: Notify,
    wake_network: Notify,
    // The jobs currently running, at most one per lane. A `Vec` rather than a map
    // because it holds two entries at the very most and every read is "find the one
    // with this id".
    live: Mutex<Vec<Arc<Live>>>,
    // The board-level change feed. Two consumers: the /jobs page, which refetches
    // when it sees a change, and a chat socket waiting for its own job to go live
    // so it can attach to the token stream without polling for it.
    events: broadcast::Sender<String>,
}

impl Queue {
    pub fn new(pool: PgPool) -> Queue {
        let (events, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        Queue {
            pool,
            wake_device: Notify::new(),
            wake_network: Notify::new(),
            live: Mutex::new(Vec::new()),
            events,
        }
    }

    // Signals both runners that the queue may have work. Safe to call spuriously,
    // and cheaper to wake both than to make every caller work out which lane the
    // job it just touched belongs to.
    pub fn wake(&self) {
        self.wake_device.notify_one();
        self.wake_network.notify_one();
    }

    // The live handles for one specific job, or None if it isn't currently running.
    pub fn live_for(&self, id_job: Uuid) -> Option<Arc<Live>> {
        lock(&self.live)
            .iter()
            .find(|l| l.id_job == id_job)
            .cloned()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<String> {
        self.events.subscribe()
    }

    // Asks a running job to stop. The worker notices at its next step boundary;
    // `false` means this job isn't running (queued, or already finished).
    pub fn request_cancel(&self, id_job: Uuid) -> bool {
        match self.live_for(id_job) {
            Some(live) => {
                live.cancel.store(true, Ordering::SeqCst);
                true
            }
            None => false,
        }
    }

    // Announces that the board changed. The payload is deliberately just a nudge:
    // the client refetches the list, which keeps one description of a job (the
    // REST shape) instead of two that could drift.
    pub fn notify_changed(&self) {
        let _ = self
            .events
            .send(serde_json::json!({ "type": "changed" }).to_string());
    }

    // Progress, unlike every other board event, carries its payload rather than
    // nudging a refetch.
    //
    // The nudge-and-refetch shape exists so a job has exactly one description; it
    // works because the board only changes when a job is queued, starts or ends —
    // rarely. Progress moves continuously, and refetching the whole board every
    // couple of seconds to advance two scalars on one row is the wrong trade. This
    // patches those two fields on a row the client already has, so there is still
    // only one description of a job; this just amends part of it.
    //
    // Emitted on the same throttle as the durable write, so the board and the row
    // never disagree about how far along a job is.
    fn notify_progress(&self, id_job: Uuid, fraction: f64, stage: &str) {
        let _ = self.events.send(
            serde_json::json!({
                "type": "progress",
                "id_job": id_job,
                "progress": fraction,
                "stage": stage,
            })
            .to_string(),
        );
    }

    // ── the loop ────────────────────────────────────────────────────────────

    // One runner. Spawned once per lane, and identical either way — the lane only
    // decides which rows it is allowed to claim.
    //
    // Pause stops both lanes. It reads as "stop starting work", not "stop using the
    // GPU", and a pause that left a lane running would be a surprising thing to
    // discover from the board.
    pub async fn run(self: Arc<Self>, lane: JobLane) {
        loop {
            match db::job::is_paused(&self.pool).await {
                Ok(true) => {
                    self.idle(lane).await;
                    continue;
                }
                Ok(false) => {}
                Err(e) => {
                    eprintln!("[queue] could not read pause state: {}", e);
                    self.idle(lane).await;
                    continue;
                }
            }

            match db::job::claim_next(&self.pool, lane).await {
                Ok(Some(job)) => self.run_one(job).await,
                Ok(None) => self.idle(lane).await,
                Err(e) => {
                    eprintln!("[queue] could not claim a job: {}", e);
                    self.idle(lane).await;
                }
            }
        }
    }

    // Waits to be woken, with a fallback tick so a missed signal can't wedge the
    // queue.
    async fn idle(&self, lane: JobLane) {
        let wake = match lane {
            JobLane::Device => &self.wake_device,
            JobLane::Network => &self.wake_network,
        };
        let _ = tokio::time::timeout(IDLE_POLL, wake.notified()).await;
    }

    // Runs one claimed job start to finish, then closes out its row.
    async fn run_one(self: &Arc<Self>, job: db::job::Job) {
        let id_job = job.id_job;
        eprintln!("[queue] starting {} job {}", kind_name(job.kind), id_job);

        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let (progress_tx, progress_rx) = mpsc::unbounded_channel::<ProgressTick>();

        // Both consumers of the thumbs feedback read files, so refresh them from the
        // table before either kind starts. Done here, on the async side that owns
        // the pool, because the workers below are blocking CUDA threads that must
        // never touch Postgres.
        //
        // Cheap enough to run unconditionally (one query, two small writes) and
        // safer than gating on the training mode: a KTO run that found a stale
        // export would silently tune on yesterday's feedback.
        // Auto-labelling is in this list because its first stage calibrates the
        // judge against the human ratings — reading a stale export would fit the
        // threshold to yesterday's thumbs and silently ignore the ones just given.
        if matches!(
            job.kind,
            JobKind::Train | JobKind::Benchmark | JobKind::Autolabel
        ) {
            if let Err(e) = export_preference_data(&self.pool).await {
                // Not fatal for either kind. A benchmark reports "no feedback" and
                // scores every other stage; a KTO run fails later, in its dataset
                // build, with a message about what it actually found.
                eprintln!("[queue] could not export preference feedback: {}", e);
            }
        }

        let done = match job.kind {
            JobKind::Train => {
                self.run_train(&job, cancel.clone(), tx, progress_tx, progress_rx)
                    .await
            }
            JobKind::Benchmark => {
                self.run_benchmark(&job, cancel.clone(), tx, progress_tx, progress_rx)
                    .await
            }
            JobKind::Chat => {
                self.run_chat(&job, cancel.clone(), tx, progress_tx, progress_rx)
                    .await
            }
            JobKind::Autolabel => {
                self.run_autolabel(&job, cancel.clone(), tx, progress_tx, progress_rx)
                    .await
            }
            JobKind::Gather => {
                self.run_gather(&job, cancel.clone(), tx, progress_tx, progress_rx)
                    .await
            }
            JobKind::Synth => {
                self.run_synth(&job, cancel.clone(), tx, progress_tx, progress_rx)
                    .await
            }
        };

        // Released before the terminal write, so a client that reacts to the
        // `changed` event never sees a finished row still claiming a runner.
        lock(&self.live).retain(|l| l.id_job != id_job);

        if let Err(e) = db::job::finish(
            &self.pool,
            id_job,
            done.status,
            done.progress,
            done.ref_id,
            done.error.as_deref(),
        )
        .await
        {
            // The row stays `running` with no worker behind it. The next restart's
            // `fail_orphaned` is the backstop; log loudly because until then this
            // job holds the head of the queue.
            eprintln!("[queue] could not close out job {}: {}", id_job, e);
        }
        eprintln!("[queue] finished job {}", id_job);
        self.notify_changed();
    }

    // Publishes the live handles and tells any waiting socket the job has started.
    // Done inside each `run_*` rather than in `run_one` because the per-kind async
    // preparation (loading a model, reading history) happens first and can fail
    // before there is anything to attach to.
    fn go_live(&self, id_job: Uuid, kind: JobKind, live: Live) {
        {
            let mut held = lock(&self.live);
            // Defensive: a lane only ever runs one job, so the id cannot already be
            // here — but an entry left behind would shadow the new one for every
            // socket that attaches.
            held.retain(|l| l.id_job != id_job);
            held.push(Arc::new(live));
        }
        let _ = self.events.send(
            serde_json::json!({
                "type": "started",
                "id_job": id_job,
                "kind": kind,
            })
            .to_string(),
        );
        self.notify_changed();
    }

    // ── per-kind execution ──────────────────────────────────────────────────

    async fn run_train(
        self: &Arc<Self>,
        job: &db::job::Job,
        cancel: Arc<AtomicBool>,
        tx: broadcast::Sender<String>,
        progress_tx: mpsc::UnboundedSender<ProgressTick>,
        progress_rx: mpsc::UnboundedReceiver<ProgressTick>,
    ) -> Done {
        let config: crate::training::TrainConfig = match serde_json::from_value(job.config.clone()) {
            Ok(config) => config,
            // Only reachable if the row was written by an older build or edited by
            // hand — the enqueue route validates and serializes it.
            Err(e) => return Done::failed(format!("stored training config is unreadable: {}", e)),
        };

        let ctx = Arc::new(TrainCtx::new(
            config.clone(),
            cancel,
            tx.clone(),
            progress_tx,
        ));
        self.go_live(job.id_job, JobKind::Train, Live {
            id_job: job.id_job,
            cancel: ctx.cancel.clone(),
            tx,
            snapshot: Snapshot::Train(ctx.clone()),
        });

        let worker_ctx = ctx.clone();
        let outcome = self
            .spawn_worker(job.id_job, progress_rx, move || {
                train_worker::run(worker_ctx, config)
            })
            .await;

        match outcome {
            Ok(train_worker::Outcome::Done { model }) => {
                let id_model = model.id_model;
                // The registry insert is what makes the model selectable in chat.
                // A failure here is a real failure of the job: the weights exist on
                // disk but nothing can find them.
                if let Err(e) = db::model::upsert(&self.pool, &model).await {
                    eprintln!("[queue] could not register model {}: {}", model.name, e);
                    return Done::failed("training finished but the model could not be registered".to_string());
                }
                Done {
                    status: JobStatus::Done,
                    progress: 1.0,
                    ref_id: Some(id_model),
                    error: None,
                }
            }
            Ok(train_worker::Outcome::Cancelled) => Done::cancelled(fraction_of(&ctx)),
            Ok(train_worker::Outcome::Failed(msg)) => Done::failed(msg),
            Err(panic) => {
                // The worker died mid-run, so nothing updated the phase; do it here
                // or a reattaching socket would see a run stuck "training".
                mark_train_crashed(&ctx, &panic);
                Done::failed(panic)
            }
        }
    }

    async fn run_benchmark(
        self: &Arc<Self>,
        job: &db::job::Job,
        cancel: Arc<AtomicBool>,
        tx: broadcast::Sender<String>,
        progress_tx: mpsc::UnboundedSender<ProgressTick>,
        progress_rx: mpsc::UnboundedReceiver<ProgressTick>,
    ) -> Done {
        let id_model = match job.config.get("id_model").and_then(|v| v.as_str()) {
            Some(id) => match Uuid::parse_str(id) {
                Ok(id) => id,
                Err(_) => return Done::failed("stored benchmark config names an invalid model".to_string()),
            },
            None => return Done::failed("stored benchmark config names no model".to_string()),
        };

        // Resolved now rather than at enqueue: a model can be deleted while its
        // benchmark waits in line, and that should fail the job with a legible
        // reason rather than crash a worker.
        let model = match db::model::find_by_id(&self.pool, id_model).await {
            Ok(Some(model)) => model,
            Ok(None) => return Done::failed("the model to benchmark no longer exists".to_string()),
            Err(e) => {
                eprintln!("[queue] benchmark model lookup failed: {}", e);
                return Done::failed("could not load the model to benchmark".to_string());
            }
        };
        let model_dir = storage::models_dir().join(id_model.to_string());
        if !model_dir.join("merges.bin").exists() {
            return Done::failed("the model has no tokenizer on disk; it cannot be benchmarked".to_string());
        }

        // The history row is opened here, not at enqueue, so a benchmark waiting in
        // line doesn't sit in the benchmark history showing "running" for hours.
        let id_run = Uuid::new_v4();
        let train_tokens = db::model::train_tokens(&model);
        if let Err(e) =
            db::benchmark::create(&self.pool, id_run, id_model, &model.name, train_tokens).await
        {
            eprintln!("[queue] could not open benchmark run: {}", e);
            return Done::failed("could not record the benchmark run".to_string());
        }

        let state = Arc::new(Mutex::new(SimpleState {
            stage: "Loading model".to_string(),
            fraction: 0.0,
            text: String::new(),
        }));
        self.go_live(job.id_job, JobKind::Benchmark, Live {
            id_job: job.id_job,
            cancel: cancel.clone(),
            tx: tx.clone(),
            snapshot: Snapshot::Simple(state.clone()),
        });

        let reporting = state.clone();
        let frame_tx = tx.clone();
        let worker_cancel = cancel.clone();
        let outcome = self
            .spawn_worker(job.id_job, progress_rx, move || {
                let config = llm::eval::EvalConfig::new(model_dir, &storage::eval_dir());
                let on_progress: llm::eval::ProgressCallback = Box::new(move |stage, fraction| {
                    {
                        let mut st = lock(&reporting);
                        st.stage = stage.to_string();
                        st.fraction = fraction;
                    }
                    let _ = progress_tx.send(ProgressTick {
                        fraction,
                        stage: stage.to_string(),
                    });
                    emit(&frame_tx, frame(
                        "progress",
                        serde_json::json!({ "stage": stage, "fraction": fraction }),
                    ));
                });
                llm::eval::run(&config, on_progress, Some(worker_cancel))
            })
            .await;

        let fraction = lock(&state).fraction;
        match outcome {
            Ok(Ok(Some(report))) => {
                // The composite score is NaN when nothing could be measured, and
                // `to_value` renders that as null — so reading the score back out
                // of the serialized form yields exactly the verdict the report
                // shows, rather than a number that would plot as a real zero.
                let value = match serde_json::to_value(&report) {
                    Ok(value) => value,
                    Err(e) => {
                        let msg = format!("could not serialize the report: {}", e);
                        self.fail_benchmark(id_run, &msg).await;
                        return Done::failed(msg);
                    }
                };
                let score = value.get("score").and_then(|s| s.as_f64());
                if let Err(e) = db::benchmark::complete(&self.pool, id_run, score, &value).await {
                    eprintln!("[queue] could not store benchmark report: {}", e);
                    return Done::failed("the benchmark finished but its report could not be stored".to_string());
                }
                emit(&tx, frame("done", serde_json::json!({ "id_run": id_run })));
                Done {
                    status: JobStatus::Done,
                    progress: 1.0,
                    ref_id: Some(id_run),
                    error: None,
                }
            }
            // Cancelled partway. The run row is closed as failed rather than left
            // open: it produced no score, and a `running` row would be reopened as
            // "interrupted by a server restart" on the next boot.
            Ok(Ok(None)) => {
                self.fail_benchmark(id_run, "cancelled").await;
                Done {
                    status: JobStatus::Cancelled,
                    progress: fraction,
                    ref_id: Some(id_run),
                    error: None,
                }
            }
            Ok(Err(e)) => {
                self.fail_benchmark(id_run, &e).await;
                Done::failed(e)
            }
            Err(panic) => {
                self.fail_benchmark(id_run, &panic).await;
                Done::failed(panic)
            }
        }
    }

    // Generates candidate replies for a set of prompts, scores them with the
    // rule-based judge, and appends the labelled pairs to the preference *training*
    // data. Produces no model and no database record beyond the job row itself —
    // its whole output is a JSONL file the next KTO run will read.
    async fn run_autolabel(
        self: &Arc<Self>,
        job: &db::job::Job,
        cancel: Arc<AtomicBool>,
        tx: broadcast::Sender<String>,
        progress_tx: mpsc::UnboundedSender<ProgressTick>,
        progress_rx: mpsc::UnboundedReceiver<ProgressTick>,
    ) -> Done {
        let request: autolabel::AutolabelRequest = match serde_json::from_value(job.config.clone())
        {
            Ok(request) => request,
            Err(e) => return Done::failed(format!("stored autolabel config is unreadable: {e}")),
        };

        // Resolved now rather than at enqueue: a model can be deleted while its job
        // waits in line.
        let model_dir = storage::models_dir().join(request.id_model.to_string());
        if !model_dir.join("merges.bin").exists() {
            return Done::failed(
                "the model has no tokenizer on disk; it cannot generate".to_string(),
            );
        }

        // Prompts come from the real user turns people have actually sent — the
        // distribution the model is going to be used on. Reading them here is the
        // only database work this job does; the worker below is a blocking CUDA
        // thread and never touches the pool.
        //
        // Routed through `gather::choose_prompts` rather than used raw, for the two
        // reasons that function documents: stored turns include follow-ups that lost
        // their antecedent and prose fragments that are not questions at all, and a
        // history too thin to fill the batch should top up from the seed list rather
        // than run short. Both were true here — the catalogue this replaced was
        // generated substantially against sentence stubs.
        let history = match db::message::distinct_user_prompts(&self.pool, request.max_prompts)
            .await
        {
            Ok(prompts) => prompts,
            Err(e) => {
                eprintln!("[queue] autolabel prompt lookup failed: {e}");
                return Done::failed("could not load prompts to generate from".to_string());
            }
        };
        let prompts = gather::choose_prompts(history, request.max_prompts as usize, 0);
        if prompts.is_empty() {
            return Done::failed(
                "no usable prompts to generate from — send some chat turns first".to_string(),
            );
        }

        let state = Arc::new(Mutex::new(SimpleState {
            stage: "Loading model".to_string(),
            fraction: 0.0,
            text: String::new(),
        }));
        self.go_live(job.id_job, JobKind::Autolabel, Live {
            id_job: job.id_job,
            cancel: cancel.clone(),
            tx: tx.clone(),
            snapshot: Snapshot::Simple(state.clone()),
        });

        let reporting = state.clone();
        let frame_tx = tx.clone();
        // One clone per closure: each is `move`d into its own callback, and they
        // outlive the block that creates them.
        let frame_tx_calibration = tx.clone();
        let frame_tx_verdict = tx.clone();
        let worker_cancel = cancel.clone();
        let outcome = self
            .spawn_worker(job.id_job, progress_rx, move || {
                let on_progress: autolabel::ProgressCallback = Box::new(move |stage, fraction| {
                    {
                        let mut st = lock(&reporting);
                        st.stage = stage.to_string();
                        st.fraction = fraction;
                    }
                    let _ = progress_tx.send(ProgressTick {
                        fraction,
                        stage: stage.to_string(),
                    });
                    emit(&frame_tx, frame(
                        "progress",
                        serde_json::json!({ "stage": stage, "fraction": fraction }),
                    ));
                });
                // The calibration lands once, up front — it decides whether the
                // rest of the run means anything, so the page should not have to
                // wait until the end to show it.
                let calibration_tx = frame_tx_calibration;
                let on_calibration: autolabel::CalibrationCallback =
                    Box::new(move |calibration| {
                        emit(
                            &calibration_tx,
                            frame("calibration", serde_json::json!(calibration)),
                        );
                    });

                // One frame per verdict. Carries the text as well as the score
                // because "what did it actually generate" is the question the
                // numbers cannot answer — a catalogue can be perfectly balanced
                // and still be six hundred variations of one bad sentence.
                let verdict_tx = frame_tx_verdict;
                let on_verdict: autolabel::VerdictCallback = Box::new(move |prompt, verdict| {
                    emit(
                        &verdict_tx,
                        frame(
                            "verdict",
                            serde_json::json!({
                                "prompt": prompt,
                                "response": verdict.text,
                                "good": verdict.good,
                                "score": verdict.score,
                            }),
                        ),
                    );
                });

                autolabel::run(
                    &model_dir,
                    &prompts,
                    &request,
                    on_progress,
                    on_calibration,
                    on_verdict,
                    Some(worker_cancel),
                )
            })
            .await;

        let fraction = lock(&state).fraction;
        match outcome {
            Ok(Ok(Some(stats))) => {
                emit(&tx, frame("autolabel", serde_json::json!(stats)));
                Done::done()
            }
            Ok(Ok(None)) => Done::cancelled(fraction),
            Ok(Err(e)) => Done::failed(e),
            Err(panic) => Done::failed(panic),
        }
    }

    // Generates a batch of short replies for a person to rate. The autolabel twin
    // with the judge replaced by a human — so unlike autolabel it writes no file:
    // the replies land in `signal_samples` unrated, and only a rated one becomes
    // training data.
    async fn run_gather(
        self: &Arc<Self>,
        job: &db::job::Job,
        cancel: Arc<AtomicBool>,
        tx: broadcast::Sender<String>,
        progress_tx: mpsc::UnboundedSender<ProgressTick>,
        progress_rx: mpsc::UnboundedReceiver<ProgressTick>,
    ) -> Done {
        let request: gather::GatherRequest = match serde_json::from_value(job.config.clone()) {
            Ok(request) => request,
            Err(e) => return Done::failed(format!("stored gather config is unreadable: {e}")),
        };

        // Resolved now rather than at enqueue: a model can be deleted while its job
        // waits in line.
        let model_dir = storage::models_dir().join(request.id_model.to_string());
        if !model_dir.join("merges.bin").exists() {
            return Done::failed(
                "the model has no tokenizer on disk; it cannot generate".to_string(),
            );
        }

        // Real prompts where they exist, seeds to top up. Both database reads happen
        // here, on the async side that owns the pool; the worker below is a blocking
        // CUDA thread and never touches it.
        let batch = request.batch.clamp(1, 64);
        let history = match db::message::distinct_user_prompts(&self.pool, batch).await {
            Ok(prompts) => prompts,
            Err(e) => {
                eprintln!("[queue] gather prompt lookup failed: {e}");
                return Done::failed("could not load prompts to generate from".to_string());
            }
        };
        // Rotated by how much has been gathered already, so a second batch on an
        // empty history asks different questions than the first rather than
        // regenerating the same list of seeds.
        let seed_offset = match db::signal::counts(&self.pool).await {
            Ok(counts) => (counts.desirable + counts.undesirable + counts.pending) as usize,
            // Only picks the starting question; a failed count is not worth failing
            // the run over.
            Err(_) => 0,
        };
        let prompts = gather::choose_prompts(history, batch as usize, seed_offset);
        if prompts.is_empty() {
            return Done::failed("no prompts to generate from".to_string());
        }

        let state = Arc::new(Mutex::new(SimpleState {
            stage: "Loading model".to_string(),
            fraction: 0.0,
            text: String::new(),
        }));
        self.go_live(job.id_job, JobKind::Gather, Live {
            id_job: job.id_job,
            cancel: cancel.clone(),
            tx: tx.clone(),
            snapshot: Snapshot::Simple(state.clone()),
        });

        // Samples cross back one at a time so the panel can be rating the third
        // reply while the fourth generates — the whole reason this is a stream and
        // not a batch handed over at the end. The writer task holds the pool; the
        // worker only sends down the channel.
        let (sample_tx, mut sample_rx) = mpsc::unbounded_channel::<(String, String)>();
        let writer_pool = self.pool.clone();
        let writer_tx = tx.clone();
        let id_job = job.id_job;
        let id_model = request.id_model;
        let writer = tokio::spawn(async move {
            while let Some((prompt, response)) = sample_rx.recv().await {
                match db::signal::insert(&writer_pool, id_job, id_model, &prompt, &response).await {
                    Ok(sample) => emit(&writer_tx, frame("sample", serde_json::json!({
                        "id_sample": sample.id_sample,
                        "prompt": sample.prompt,
                        "response": sample.response,
                    }))),
                    // The reply is lost, but the run isn't: the rest of the batch is
                    // still worth generating, and a sample that was never stored is
                    // one the panel simply never offers.
                    Err(e) => eprintln!("[queue] could not store a gathered sample: {e}"),
                }
            }
        });

        let reporting = state.clone();
        let frame_tx = tx.clone();
        let worker_cancel = cancel.clone();
        let outcome = self
            .spawn_worker(job.id_job, progress_rx, move || {
                let on_progress: gather::ProgressCallback = Box::new(move |stage, fraction| {
                    {
                        let mut st = lock(&reporting);
                        st.stage = stage.to_string();
                        st.fraction = fraction;
                    }
                    let _ = progress_tx.send(ProgressTick {
                        fraction,
                        stage: stage.to_string(),
                    });
                    emit(&frame_tx, frame(
                        "progress",
                        serde_json::json!({ "stage": stage, "fraction": fraction }),
                    ));
                });
                let on_sample: gather::SampleCallback = Box::new(move |prompt, response| {
                    // Fails only if the writer task is gone, i.e. at shutdown.
                    let _ = sample_tx.send((prompt.to_string(), response.to_string()));
                });
                gather::run(
                    &model_dir,
                    &prompts,
                    &request,
                    on_progress,
                    on_sample,
                    Some(worker_cancel),
                )
            })
            .await;

        // The worker is finished, so its sender is dropped and the writer drains and
        // exits. Awaited before the terminal row is written, so a client that reacts
        // to the finished job never fetches the batch before its last row is in.
        let _ = writer.await;

        let fraction = lock(&state).fraction;
        match outcome {
            Ok(Ok(Some(stats))) => {
                emit(&tx, frame("gather", serde_json::json!(stats)));
                Done::done()
            }
            Ok(Ok(None)) => Done::cancelled(fraction),
            Ok(Err(e)) => Done::failed(e),
            Err(panic) => Done::failed(panic),
        }
    }

    async fn fail_benchmark(&self, id_run: Uuid, error: &str) {
        if let Err(e) = db::benchmark::fail(&self.pool, id_run, error).await {
            eprintln!("[queue] could not close out benchmark run {}: {}", id_run, e);
        }
    }

    async fn run_chat(
        self: &Arc<Self>,
        job: &db::job::Job,
        cancel: Arc<AtomicBool>,
        tx: broadcast::Sender<String>,
        progress_tx: mpsc::UnboundedSender<ProgressTick>,
        progress_rx: mpsc::UnboundedReceiver<ProgressTick>,
    ) -> Done {
        let request: ChatRequest = match serde_json::from_value(job.config.clone()) {
            Ok(request) => request,
            Err(e) => return Done::failed(format!("stored chat request is unreadable: {}", e)),
        };

        let model_dir = storage::models_dir().join(request.id_model.to_string());

        // Loaded at run time, not enqueue: by the time a queued turn reaches the
        // GPU the conversation may have grown, and the model should see the
        // conversation as it stands now — including the user message the enqueuing
        // socket persisted.
        let history = match db::message::list_for_chat(&self.pool, request.id_chat).await {
            Ok(messages) => messages
                .into_iter()
                .map(|m| llm::chat_template::Turn {
                    // id_user set => a person sent it; otherwise an assistant reply.
                    role: if m.id_user.is_some() {
                        llm::chat_template::Role::User
                    } else {
                        llm::chat_template::Role::Assistant
                    },
                    text: m.content,
                })
                .collect::<Vec<_>>(),
            Err(e) => {
                eprintln!("[queue] could not load chat history: {}", e);
                return Done::failed("could not load the conversation".to_string());
            }
        };

        let state = Arc::new(Mutex::new(SimpleState {
            stage: "Generating".to_string(),
            fraction: 0.0,
            text: String::new(),
        }));
        self.go_live(job.id_job, JobKind::Chat, Live {
            id_job: job.id_job,
            cancel: cancel.clone(),
            tx: tx.clone(),
            snapshot: Snapshot::Simple(state.clone()),
        });

        let accumulating = state.clone();
        let frame_tx = tx.clone();
        let worker_cancel = cancel.clone();
        let system = request.system.clone();
        let outcome = self
            .spawn_worker(job.id_job, progress_rx, move || {
                // A chat turn has no measurable total, so it reports one tick to
                // put a stage on the row and then streams tokens instead.
                let _ = progress_tx.send(ProgressTick {
                    fraction: 0.0,
                    stage: "Generating".to_string(),
                });
                llm::infer::generate_chat_response_streaming(
                    &model_dir,
                    system.as_deref(),
                    &history,
                    MAX_TOKENS,
                    TEMPERATURE,
                    TOP_K,
                    Some(&worker_cancel),
                    |token| {
                        lock(&accumulating).text.push_str(token);
                        emit(&frame_tx, frame("token", serde_json::json!({ "value": token })));
                    },
                )
            })
            .await;

        let was_cancelled = cancel.load(Ordering::SeqCst);
        let reply = match outcome {
            Ok(Ok(reply)) => reply,
            Ok(Err(e)) => {
                eprintln!("[queue] inference failed: {}", e);
                emit(&tx, frame("error", serde_json::json!({ "message": "inference failed" })));
                return Done::failed("inference failed".to_string());
            }
            Err(panic) => {
                eprintln!("[queue] inference crashed: {}", panic);
                emit(&tx, frame("error", serde_json::json!({ "message": "inference failed" })));
                return Done::failed(panic);
            }
        };

        // A cancelled turn still persists what it produced: the user watched those
        // tokens arrive, and dropping them would make the conversation disagree
        // with what was on screen. An empty reply is not worth a row.
        if reply.text.trim().is_empty() {
            emit(&tx, frame("done", serde_json::json!({ "id_message": serde_json::Value::Null })));
            return Done {
                status: if was_cancelled { JobStatus::Cancelled } else { JobStatus::Done },
                progress: 1.0,
                ref_id: None,
                error: None,
            };
        }

        let assistant = match db::message::create_assistant_message(
            &self.pool,
            request.id_chat,
            Some(request.id_model),
            &reply.text,
            reply.output_tokens as i32,
        )
        .await
        {
            Ok(message) => message,
            Err(e) => {
                eprintln!("[queue] could not save the reply: {}", e);
                emit(&tx, frame("error", serde_json::json!({ "message": "failed to save response" })));
                return Done::failed("could not save the reply".to_string());
            }
        };

        // Non-fatal: a tracking failure shouldn't sink a reply the user already
        // received and we already persisted.
        if let Some(id_user) = request.id_user {
            if let Err(e) = db::usage::record(
                &self.pool,
                id_user,
                Some(request.id_model),
                reply.input_tokens as i32,
                reply.output_tokens as i32,
            )
            .await
            {
                eprintln!("[queue] could not record usage: {}", e);
            }
        }

        emit(&tx, frame(
            "done",
            serde_json::json!({
                "chat_id": request.id_chat,
                "id_message": assistant.id_message,
            }),
        ));
        Done {
            status: if was_cancelled { JobStatus::Cancelled } else { JobStatus::Done },
            progress: 1.0,
            ref_id: Some(assistant.id_message),
            error: None,
        }
    }

    // Asks a cheap hosted model for synthetic SFT conversations, validates them
    // against the real tag vocabulary and the base model's context window, and
    // appends the survivors to the SFT corpus. Produces no database record beyond
    // its own row — its whole output is a JSONL file the next SFT run will read.
    //
    // The one kind with no worker thread. Every other job here is blocking CUDA and
    // has to leave the async runtime to avoid stalling it; this is HTTP and parsing,
    // so it runs as an ordinary future on the runner's own task. `spawn_worker`
    // would have to block a whole OS thread on a network wait to achieve nothing.
    async fn run_synth(
        self: &Arc<Self>,
        job: &db::job::Job,
        cancel: Arc<AtomicBool>,
        tx: broadcast::Sender<String>,
        progress_tx: mpsc::UnboundedSender<ProgressTick>,
        mut progress_rx: mpsc::UnboundedReceiver<ProgressTick>,
    ) -> Done {
        let request: synth::SynthRequest = match serde_json::from_value(job.config.clone()) {
            Ok(request) => request,
            Err(e) => return Done::failed(format!("stored generation config is unreadable: {e}")),
        };

        let state = Arc::new(Mutex::new(SynthState {
            stage: "Starting".to_string(),
            fraction: 0.0,
            calls: Vec::new(),
            accepted: Vec::new(),
            added: 0,
            target: request.count,
            duplicates: 0,
            with_thinking: 0,
        }));
        self.go_live(job.id_job, JobKind::Synth, Live {
            id_job: job.id_job,
            cancel: cancel.clone(),
            tx: tx.clone(),
            snapshot: Snapshot::Synth(state.clone()),
        });

        // One clone per callback: each is `move`d into its own closure and they all
        // outlive this block.
        let progress_state = state.clone();
        let progress_frames = tx.clone();
        let on_progress: synth::ProgressCallback = Box::new(move |stage, fraction| {
            {
                let mut st = lock(&progress_state);
                st.stage = stage.to_string();
                st.fraction = fraction;
            }
            let _ = progress_tx.send(ProgressTick {
                fraction,
                stage: stage.to_string(),
            });
            emit(&progress_frames, frame(
                "progress",
                serde_json::json!({ "stage": stage, "fraction": fraction }),
            ));
        });

        // One frame per model call. This is the frame that makes a run diagnosable:
        // a provider erroring, a model whose output is all rejected for one reason,
        // and a model repeating itself are three different failures that a progress
        // bar renders identically.
        let call_state = state.clone();
        let call_frames = tx.clone();
        let on_call: synth::CallCallback = Box::new(move |report| {
            {
                let mut st = lock(&call_state);
                st.calls.push(report.clone());
                st.added = report.total;
                st.duplicates += report.duplicates;
            }
            emit(&call_frames, frame("call", serde_json::json!(report)));
        });

        let accepted_state = state.clone();
        let accepted_frames = tx.clone();
        let on_accepted: synth::AcceptedCallback = Box::new(move |accepted| {
            {
                let mut st = lock(&accepted_state);
                if accepted.think_words > 0 {
                    st.with_thinking += 1;
                }
                st.accepted.insert(0, accepted.clone());
                st.accepted.truncate(SYNTH_FEED);
            }
            emit(&accepted_frames, frame("accepted", serde_json::json!(accepted)));
        });

        // Progress is drained concurrently with the run rather than through
        // `spawn_worker`, which only knows how to await a thread. Same throttle, same
        // reason: the frames carry the truth, this is so a fresh page load has a
        // sane bar before its socket attaches.
        let run = synth::run(&request, on_progress, on_call, on_accepted, cancel.clone());
        tokio::pin!(run);
        let mut last_write = Instant::now() - PROGRESS_WRITE_INTERVAL;
        let outcome = loop {
            tokio::select! {
                biased;
                result = &mut run => break result,
                Some(tick) = progress_rx.recv() => {
                    if last_write.elapsed() >= PROGRESS_WRITE_INTERVAL {
                        last_write = Instant::now();
                        if let Err(e) =
                            db::job::set_progress(&self.pool, job.id_job, tick.fraction, &tick.stage).await
                        {
                            eprintln!("[queue] could not write progress for {}: {}", job.id_job, e);
                        }
                        self.notify_progress(job.id_job, tick.fraction, &tick.stage);
                    }
                }
            }
        };

        let fraction = lock(&state).fraction;
        match outcome {
            Ok(Some(stats)) => {
                emit(&tx, frame("synth", serde_json::json!(stats)));
                Done::done()
            }
            // Cancelled runs keep what they wrote — the file is appended to as each
            // conversation is accepted — so this is a partial success reported as a
            // cancellation, not a loss.
            Ok(None) => Done::cancelled(fraction),
            Err(e) => Done::failed(e),
        }
    }

    // ── worker plumbing ─────────────────────────────────────────────────────

    // Runs `work` on its own OS thread and awaits the result, writing coarse
    // progress to the durable row on a throttle in the meantime.
    //
    // Every kind of job here is FFI into CUDA, where a device fault or a failed
    // invariant arrives as a panic. On a bare thread that would die silently and
    // leave the row `running` with nothing behind it, so the panic is caught and
    // returned as an `Err` the queue can report.
    async fn spawn_worker<T, F>(
        &self,
        id_job: Uuid,
        mut progress_rx: mpsc::UnboundedReceiver<ProgressTick>,
        work: F,
    ) -> Result<T, String>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (done_tx, mut done_rx) = oneshot::channel();
        std::thread::spawn(move || {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
            // Fails only if the runner is gone, i.e. at shutdown.
            let _ = done_tx.send(outcome.map_err(|payload| panic_message(&*payload)));
        });

        // Seeded in the past so the first report lands immediately rather than
        // after a blank interval.
        let mut last_write = Instant::now() - PROGRESS_WRITE_INTERVAL;
        loop {
            tokio::select! {
                // Biased so a finished run is noticed before draining more
                // progress: the terminal write should not queue behind ticks.
                biased;
                result = &mut done_rx => {
                    return match result {
                        Ok(outcome) => outcome,
                        Err(_) => Err("the worker thread vanished".to_string()),
                    };
                }
                Some(tick) = progress_rx.recv() => {
                    if last_write.elapsed() >= PROGRESS_WRITE_INTERVAL {
                        last_write = Instant::now();
                        if let Err(e) =
                            db::job::set_progress(&self.pool, id_job, tick.fraction, &tick.stage).await
                        {
                            // Cosmetic: the live frames still carry the truth.
                            eprintln!("[queue] could not write progress for {}: {}", id_job, e);
                        }
                        self.notify_progress(id_job, tick.fraction, &tick.stage);
                    }
                }
            }
        }
    }
}

// What a run produced, before it becomes a row.
pub struct Done {
    pub status: JobStatus,
    pub progress: f64,
    pub ref_id: Option<Uuid>,
    pub error: Option<String>,
}

impl Done {
    fn failed(error: String) -> Done {
        Done {
            status: JobStatus::Failed,
            progress: 0.0,
            ref_id: None,
            error: Some(error),
        }
    }

    fn cancelled(progress: f64) -> Done {
        Done {
            status: JobStatus::Cancelled,
            progress,
            ref_id: None,
            error: None,
        }
    }

    // A run that finished and produced no domain record to point at. Autolabel's
    // whole output is a file, so there is nothing in `models`, `benchmark_runs` or
    // `messages` for `ref_id` to name.
    fn done() -> Done {
        Done {
            status: JobStatus::Done,
            progress: 1.0,
            ref_id: None,
            error: None,
        }
    }
}

// The chat turn a queued `chat` job carries. Written by the chat socket at enqueue
// (which has already persisted the user's message) and read back here.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ChatRequest {
    pub id_chat: Uuid,
    pub id_model: Uuid,
    // Who to bill the usage to. Optional so a row whose user was deleted still
    // runs rather than failing on a missing id.
    pub id_user: Option<Uuid>,
    pub system: Option<String>,
}

// How far a cancelled training run had got, for the terminal row.
fn fraction_of(ctx: &TrainCtx) -> f64 {
    let st = lock(&ctx.state);
    st.train
        .get("epoch")
        .and_then(|e| e.as_f64())
        .zip(st.train.get("total_epochs").and_then(|t| t.as_f64()))
        .filter(|(_, total)| *total > 0.0)
        .map(|(epoch, total)| ((epoch - 1.0) / total).clamp(0.0, 1.0))
        .unwrap_or(0.0)
}

// Records a crashed run in the state a socket reads, so a reattach reports the
// crash instead of a run that appears to still be training.
fn mark_train_crashed(ctx: &TrainCtx, message: &str) {
    eprintln!("[queue] training crashed: {}", message);
    let mut st: std::sync::MutexGuard<'_, TrainState> = lock(&ctx.state);
    if !st.phase.is_terminal() {
        st.phase = TrainPhase::Failed;
        st.error = Some(message.to_string());
        drop(st);
        emit(&ctx.tx, frame(
            "error",
            serde_json::json!({ "message": message }),
        ));
    }
}

fn kind_name(kind: JobKind) -> &'static str {
    match kind {
        JobKind::Train => "train",
        JobKind::Benchmark => "benchmark",
        JobKind::Chat => "chat",
        JobKind::Autolabel => "autolabel",
        JobKind::Gather => "gather",
        JobKind::Synth => "synth",
    }
}

// Pulls a human-readable message out of a panic payload (panics carry either a
// &str or a String).
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        format!("worker crashed: {}", s)
    } else if let Some(s) = payload.downcast_ref::<String>() {
        format!("worker crashed: {}", s)
    } else {
        "worker crashed".to_string()
    }
}

// ── small shared helpers ────────────────────────────────────────────────────

// Locks a mutex, recovering the guard if a previous holder panicked while holding
// it. The guarded sections here are single field reads/writes that don't
// themselves panic, so poisoning isn't expected — but a long-running server
// shouldn't turn one unlucky panic into a cascade of `.unwrap()` panics across
// every socket handler, so we take the last state and carry on. Centralizes the
// justification that would otherwise repeat at each `.lock()` (R3).
pub fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

// Best-effort broadcast of a live frame. `send` errors only when no socket is
// currently attached (no receivers); that's normal — the job runs regardless of
// who is listening, and a re-attaching socket catches up from the snapshot.
// Dropping the result is intentional (R3).
pub fn emit(tx: &broadcast::Sender<String>, msg: String) {
    let _ = tx.send(msg);
}

// Tags a JSON object with its frame `type` and serializes it.
pub fn frame(kind: &str, mut obj: serde_json::Value) -> String {
    if let Some(map) = obj.as_object_mut() {
        map.insert("type".into(), serde_json::Value::String(kind.to_string()));
    }
    obj.to_string()
}

// Fraction of rated replies reserved for the benchmark and never trained on.
const HELDOUT_FRACTION: u64 = 5; // 1 in 5 = 20%

// Exports every rated assistant reply as JSONL, split into a training slice
// (`raw/preference/`, read by `build_kto_dataset`) and a held-out slice
// (`eval/preference/`, read by the benchmark's preference stage). Returns
// (train, heldout) counts.
//
// This is the whole bridge between the feedback table and both consumers, and it is
// files rather than queries because the llm crate has no database awareness — the
// `eval` binary has to be runnable against a directory alone.
//
// **Why the split exists.** Preference tuning and the preference benchmark would
// otherwise read the same examples, and the margin rising after a tune would prove
// only that the model memorized the examples it was just trained on. Held-out is
// the only version of that number that means anything.
//
// **Why it splits on the prompt, not the reply.** All replies to the same prompt
// must land on the same side. The benchmark's paired accuracy compares an approved
// and a rejected reply to one prompt against each other; if the split ran per reply,
// a pair could straddle it and the tune would have seen half of a judgement the
// benchmark then scores.
//
// **Why a hash and not a shuffle.** The assignment has to be stable across runs —
// a reply that was held out last time must be held out this time, or successive
// benchmarks of the same model would score different data and the trend chart would
// be comparing two different measurements.
//
// Both files are rewritten whole rather than appended to. They are projections of
// the table, not logs: a rating can be changed or withdrawn, and an append-only file
// would keep scoring verdicts the user has since taken back.
async fn export_preference_data(pool: &PgPool) -> Result<(usize, usize), String> {
    let rated = db::message::list_rated(pool)
        .await
        .map_err(|e| format!("query failed: {e}"))?;

    // The two sources of thumbs — replies rated in a conversation, and replies
    // generated by a gather job purely to be rated — flow into one file. They are
    // the same judgement about the same kind of object, so KTO has no reason to
    // distinguish them, and the held-out split below treats them identically: it
    // hashes the prompt, so a gathered sample and a chat reply to the same question
    // still land on the same side.
    let gathered = db::signal::list_rated(pool)
        .await
        .map_err(|e| format!("query failed: {e}"))?;
    let rated: Vec<db::message::RatedReply> = rated
        .into_iter()
        .chain(gathered.into_iter().map(|s| db::message::RatedReply {
            prompt: Some(s.prompt),
            response: s.response,
            feedback: s.feedback,
        }))
        .collect();

    let train_dir = storage::preference_dir();
    let heldout_dir = storage::eval_dir().join("preference");
    for dir in [&train_dir, &heldout_dir] {
        std::fs::create_dir_all(dir).map_err(|e| format!("could not create {dir:?}: {e}"))?;
    }

    let mut train = String::new();
    let mut heldout = String::new();
    let mut train_n = 0usize;
    let mut heldout_n = 0usize;

    for reply in rated {
        // A reply with no preceding user turn has no prompt to condition on, so
        // scoring it would measure the wrong conditional. Dropped rather than
        // exported half-formed.
        let (Some(prompt), Some(feedback)) = (reply.prompt, reply.feedback) else {
            continue;
        };
        let label = match feedback {
            db::message::Feedback::Up => "up",
            db::message::Feedback::Down => "down",
        };
        // Built through serde_json rather than string-formatted, so a reply
        // containing a quote or a newline can't produce a line that won't parse back.
        let line = serde_json::json!({
            "prompt": prompt,
            "response": reply.response,
            "label": label,
        });

        if prompt_hash(prompt.trim()) % HELDOUT_FRACTION == 0 {
            heldout.push_str(&line.to_string());
            heldout.push('\n');
            heldout_n += 1;
        } else {
            train.push_str(&line.to_string());
            train.push('\n');
            train_n += 1;
        }
    }

    for (dir, body) in [(&train_dir, train), (&heldout_dir, heldout)] {
        let path = dir.join("feedback.jsonl");
        std::fs::write(&path, body).map_err(|e| format!("could not write {path:?}: {e}"))?;
    }
    Ok((train_n, heldout_n))
}

// FNV-1a over the prompt. Any stable hash would do; the standard library's is
// explicitly not stable across builds, which is exactly the property that would
// silently reshuffle the held-out set on a recompile.
fn prompt_hash(prompt: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in prompt.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

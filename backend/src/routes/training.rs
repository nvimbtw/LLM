#![allow(non_snake_case)]

use actix_web::{HttpRequest, HttpResponse, Responder, get, post, web};
use futures_util::StreamExt;
use serde::Deserialize;
use std::sync::atomic::Ordering;
use uuid::Uuid;

use crate::extractors::AdminUser;
use crate::jobs::{self, JobRegistry};
use crate::storage;
use crate::training::{TrainConfig, TrainMode};
use crate::ws;

// Kicks off a run in the background and returns its id. The actual progress is
// streamed over the /training/ws socket, attached by this id — so the run is
// decoupled from any one socket and survives a tab close / reconnect.
#[post("")]
async fn start_training(
    _admin: AdminUser,
    registry: web::Data<JobRegistry>,
    pool: web::Data<sqlx::PgPool>,
    body: web::Json<TrainConfig>,
) -> impl Responder {
    let mut config = body.into_inner();
    if let Err(msg) = validate_config(&config) {
        return HttpResponse::BadRequest().body(msg);
    }
    if let Err(msg) = validate_sft(&config) {
        return HttpResponse::BadRequest().body(msg);
    }

    // Resolve the storage id (which names the on-disk weights dir): reuse the
    // existing model's id on a retrain of the same name, else mint a new one.
    config.storage_id =
        match crate::db::model::find_id_by_name(pool.get_ref(), &config.model_name).await {
            Ok(Some(id)) => id,
            Ok(None) => uuid::Uuid::new_v4(),
            Err(e) => {
                eprintln!("resolve model id db error: {}", e);
                return HttpResponse::InternalServerError().body("failed to start training");
            }
        };

    let job_id = registry.start(config);
    HttpResponse::Created().json(serde_json::json!({ "job_id": job_id }))
}

// Lists the selectable pretrain datasets — the subfolders of uploads/, each with
// its .txt count — so the params rail can render a checkbox per dataset with an
// "n / <files>" sample cap.
#[get("/datasets")]
async fn list_datasets(_admin: AdminUser) -> impl Responder {
    match crate::training::list_datasets() {
        Ok(list) => HttpResponse::Ok().json(list),
        Err(e) => {
            eprintln!("list datasets error: {}", e);
            HttpResponse::InternalServerError().body("failed to list datasets")
        }
    }
}

// Lists known runs so the page can offer to re-attach to one already in flight.
#[get("")]
async fn list_jobs(_admin: AdminUser, registry: web::Data<JobRegistry>) -> impl Responder {
    let jobs: Vec<_> = registry
        .list()
        .into_iter()
        .map(|(job_id, status, model_name)| {
            serde_json::json!({
                "job_id": job_id,
                "status": status,
                "model_name": model_name,
            })
        })
        .collect();
    HttpResponse::Ok().json(jobs)
}

// Progress socket. Same handshake shape as the chat socket: auth in the first
// frame, then an attach frame naming the job to follow.
#[get("/ws")]
async fn training_ws(
    req: HttpRequest,
    body: web::Payload,
    registry: web::Data<JobRegistry>,
    jwt_secret: web::Data<String>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, session, msg_stream) = actix_ws::handle(&req, body)?;
    let registry = registry.clone();
    let secret = jwt_secret.get_ref().clone();
    actix_web::rt::spawn(training_session(session, msg_stream, registry, secret));
    Ok(response)
}

#[derive(Deserialize)]
struct AttachFrame {
    job_id: Uuid,
}

#[derive(Deserialize)]
struct ActionFrame {
    action: String,
}

// Drives one progress socket: authenticate, attach to a job, send a catch-up
// snapshot, then forward live frames until the run finishes or the client
// leaves. Closing the socket does NOT cancel the run — only an explicit
// {"action":"cancel"} does.
async fn training_session(
    mut session: actix_ws::Session,
    mut stream: actix_ws::MessageStream,
    registry: web::Data<JobRegistry>,
    secret: String,
) {
    use actix_ws::Message;

    // ---- authenticate on the first text frame ----
    loop {
        match stream.next().await {
            Some(Ok(Message::Text(text))) => match ws::authenticate(&text, &secret) {
                Some(_uid) => break,
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
    // Best-effort ready signal; if the peer is gone the next read ends the loop.
    let _ = session
        .text(serde_json::json!({ "type": "ready" }).to_string())
        .await;

    // ---- attach frame: { "job_id": "<uuid>" } ----
    let job = loop {
        match stream.next().await {
            Some(Ok(Message::Text(text))) => {
                // A malformed attach frame (bad JSON) is treated as "unknown
                // job": `.ok()` drops the parse error and the None flows below.
                let attached = serde_json::from_str::<AttachFrame>(&text)
                    .ok()
                    .and_then(|f| registry.get(f.job_id));
                match attached {
                    Some(job) => break job,
                    None => {
                        // Best-effort notify + close; we return either way.
                        let _ = session.text(ws::error_frame("unknown job")).await;
                        let _ = session.close(None).await;
                        return;
                    }
                }
            }
            Some(Ok(Message::Ping(bytes))) => {
                // Best-effort pong; a dead peer ends the loop on the next read.
                let _ = session.pong(&bytes).await;
            }
            Some(Ok(Message::Close(_))) | None => return,
            _ => {}
        }
    };

    // ---- catch-up snapshot, then live frames ----
    // Subscribe while holding the state lock so a frame can't slip between the
    // snapshot read and the subscription (we'd rather risk a duplicate frame,
    // which the client renders idempotently, than miss one).
    let (snapshot, terminal, mut rx) = {
        let st = jobs::lock(&job.state);
        (
            jobs::snapshot_frame(&st),
            st.status.is_terminal(),
            job.tx.subscribe(),
        )
    };
    if session.text(snapshot).await.is_err() {
        return;
    }
    // Nothing more will be broadcast for a finished run; don't leave the socket
    // blocked on a channel that will never produce again.
    if terminal {
        // Best-effort close; nothing more will be broadcast for a finished run.
        let _ = session.close(None).await;
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
                        job.cancel.store(true, Ordering::SeqCst);
                    }
                }
                Some(Ok(Message::Ping(bytes))) => {
                    // Best-effort pong; a dead peer ends the loop on the next read.
                    let _ = session.pong(&bytes).await;
                }
                Some(Ok(Message::Close(_))) | None => break,
                _ => {}
            },
            msg = rx.recv() => match msg {
                Ok(text) => {
                    // The worker's last frame for a run is "done" or "error";
                    // forward it, then close.
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
                    let snapshot = jobs::snapshot_frame(&jobs::lock(&job.state));
                    if session.text(snapshot).await.is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            },
        }
    }

    // Best-effort close on the way out; the socket may already be gone.
    let _ = session.close(None).await;
}

// Rejects obviously-bad configs before a worker is spawned. model_name becomes a
// directory under artifacts/models, so it must stay filesystem-safe.
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
    // Pretrain builds its corpus from the selected upload datasets; SFT ignores
    // them (it fine-tunes on chat data), so only require a selection for pretrain.
    // A curriculum run supplies its datasets per stage instead of the flat list.
    if cfg.mode != TrainMode::Sft {
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

// Extra validation for an SFT run (a no-op for pretrain): the base model to
// fine-tune from must be named and must resolve to an on-disk model (a
// `model.bin` under its UUID dir). A bad or missing base is a client error (400),
// not a server fault — catch it here so the worker never spawns against a
// nonexistent base. Note: the model hyperparameters (dimensions/ctx_window/…) are
// ignored for SFT — they are loaded from the base — but must still pass
// `validate_config`; the frontend mirrors the base's values into the request
// (ISSUE-0038).
fn validate_sft(cfg: &TrainConfig) -> Result<(), &'static str> {
    if cfg.mode != TrainMode::Sft {
        return Ok(());
    }
    let base_id = cfg
        .base_model_id
        .ok_or("base_model_id is required for an SFT run")?;
    let base_dir = storage::models_dir().join(base_id.to_string());
    if !base_dir.join("model.bin").is_file() {
        return Err("base_model_id does not resolve to an existing model");
    }
    Ok(())
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(training_ws);
    cfg.service(list_datasets);
    cfg.service(start_training);
    cfg.service(list_jobs);
}

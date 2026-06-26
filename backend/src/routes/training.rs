use actix_web::{HttpRequest, HttpResponse, Responder, get, post, web};
use futures_util::StreamExt;
use serde::Deserialize;
use std::sync::atomic::Ordering;
use uuid::Uuid;

use crate::extractors::AdminUser;
use crate::jobs::{self, JobRegistry};
use crate::training::TrainConfig;
use crate::ws;

// Kicks off a run in the background and returns its id. The actual progress is
// streamed over the /training/ws socket, attached by this id — so the run is
// decoupled from any one socket and survives a tab close / reconnect.
#[post("")]
async fn start_training(
    _admin: AdminUser,
    registry: web::Data<JobRegistry>,
    body: web::Json<TrainConfig>,
) -> impl Responder {
    let config = body.into_inner();
    if let Err(msg) = validate_config(&config) {
        return HttpResponse::BadRequest().body(msg);
    }
    let job_id = registry.start(config);
    HttpResponse::Created().json(serde_json::json!({ "job_id": job_id }))
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
                    let _ = session.text(ws::error_frame("authentication failed")).await;
                    let _ = session.close(None).await;
                    return;
                }
            },
            Some(Ok(Message::Ping(bytes))) => {
                let _ = session.pong(&bytes).await;
            }
            Some(Ok(Message::Close(_))) | None => return,
            _ => {}
        }
    }
    let _ = session
        .text(serde_json::json!({ "type": "ready" }).to_string())
        .await;

    // ---- attach frame: { "job_id": "<uuid>" } ----
    let job = loop {
        match stream.next().await {
            Some(Ok(Message::Text(text))) => {
                let attached = serde_json::from_str::<AttachFrame>(&text)
                    .ok()
                    .and_then(|f| registry.get(f.job_id));
                match attached {
                    Some(job) => break job,
                    None => {
                        let _ = session.text(ws::error_frame("unknown job")).await;
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

    // ---- catch-up snapshot, then live frames ----
    // Subscribe while holding the state lock so a frame can't slip between the
    // snapshot read and the subscription (we'd rather risk a duplicate frame,
    // which the client renders idempotently, than miss one).
    let (snapshot, terminal, mut rx) = {
        let st = job.state.lock().unwrap();
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
                    let snapshot = jobs::snapshot_frame(&job.state.lock().unwrap());
                    if session.text(snapshot).await.is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            },
        }
    }

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
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err("model_name may only contain letters, digits, '_' or '-'");
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
    Ok(())
}

fn is_power_of_two(n: i64) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(training_ws);
    cfg.service(start_training);
    cfg.service(list_jobs);
}

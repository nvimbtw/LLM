//! Upgrade browser sockets; session modules own authentication, readers and stream lifetimes.

mod chat;
mod connection;
mod gym;
mod jobs;
mod reader;

use actix_web::{get, web, HttpRequest, HttpResponse};
use llm_engine::Proxy;
use tools::Registry;
use uuid::Uuid;

/// Upgrade a model-job socket and start its authenticated event session.
#[get("/runs/{run_id}")]
async fn run_socket(
    req: HttpRequest,
    body: web::Payload,
    run_id: web::Path<Uuid>,
    engine: web::Data<Proxy>,
    secret: web::Data<String>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, session, stream) = actix_ws::handle(&req, body)?;
    actix_web::rt::spawn(jobs::run(
        session,
        stream,
        engine.get_ref().clone(),
        run_id.into_inner(),
        secret.get_ref().clone(),
    ));
    Ok(response)
}

/// Upgrade a data-job socket; cancellation remains an HTTP operation.
#[get("/data/runs/{run_id}")]
async fn data_run_socket(
    req: HttpRequest,
    body: web::Payload,
    run_id: web::Path<Uuid>,
    data: web::Data<data_engine::Proxy>,
    secret: web::Data<String>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, session, stream) = actix_ws::handle(&req, body)?;
    actix_web::rt::spawn(jobs::data_run(
        session,
        stream,
        data.get_ref().clone(),
        run_id.into_inner(),
        secret.get_ref().clone(),
    ));
    Ok(response)
}

/// Upgrade a gym socket; its session enforces admin authorization before driving episodes.
#[get("/gyms/runs/{run_id}")]
async fn gym_socket(
    req: HttpRequest,
    body: web::Payload,
    run_id: web::Path<Uuid>,
    engine: web::Data<Proxy>,
    gyms: web::Data<gyms_engine::Proxy>,
    secret: web::Data<String>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, session, stream) = actix_ws::handle(&req, body)?;
    actix_web::rt::spawn(gym::run(
        session,
        stream,
        engine.get_ref().clone(),
        gyms.get_ref().clone(),
        run_id.into_inner(),
        secret.get_ref().clone(),
    ));
    Ok(response)
}

/// Upgrade a conversation socket and transfer its resources to one authenticated turn session.
#[get("/chats/{chat_id}")]
async fn chat_socket(
    req: HttpRequest,
    body: web::Payload,
    chat_id: web::Path<Uuid>,
    engine: web::Data<Proxy>,
    registry: web::Data<Registry>,
    secret: web::Data<String>,
    pool: web::Data<sqlx::PgPool>,
) -> Result<HttpResponse, actix_web::Error> {
    let (response, session, stream) = actix_ws::handle(&req, body)?;
    actix_web::rt::spawn(chat::run(
        session,
        stream,
        chat::Connection {
            engine: engine.get_ref().clone(),
            registry: registry.get_ref().clone(),
            pool: pool.get_ref().clone(),
            chat_id: chat_id.into_inner(),
            secret: secret.get_ref().clone(),
        },
    ));
    Ok(response)
}

/// Mount the specific data/gym families before the generic run UUID extractor.
pub fn configure(cfg: &mut web::ServiceConfig) {
    cfg.service(data_run_socket);
    cfg.service(gym_socket);
    cfg.service(run_socket);
    cfg.service(chat_socket);
}

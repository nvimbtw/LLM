//! Durable job event sessions; observing and cancelling are separate operations.

use super::{
    connection::{handshake, relay},
    reader::{self, SocketReader},
};
use crate::ws;
use llm_engine::Proxy;
use uuid::Uuid;

/// Authenticate and observe a model job, owning its optional admin control reader for the session lifetime.
/// Disconnect or task cancellation stops observation; only an explicit cancel frame cancels the job.
pub(super) async fn run(
    mut session: actix_ws::Session,
    mut stream: actix_ws::MessageStream,
    engine: Proxy,
    run_id: Uuid,
    secret: String,
) {
    let Some(claims) = handshake(&mut session, &mut stream, &secret).await else {
        return;
    };
    let caller = auth::Caller::User(claims.sub);
    let control = SocketReader::new(actix_web::rt::spawn(reader::cancellation(
        stream,
        engine.clone(),
        run_id,
        matches!(claims.access_level, auth::AccessLevel::Admin),
    )));
    match engine
        .events(&caller, &format!("/runs/{run_id}/events"))
        .await
    {
        Ok(frames) => relay(&mut session, frames).await,
        Err(error) => {
            let _ = session.text(ws::error_frame(&error.to_string())).await;
        }
    }
    drop(control);
    let _ = session.close(None).await;
}

/// Observe data-job events after authentication; its cancellation contract remains the existing HTTP route.
pub(super) async fn data_run(
    mut session: actix_ws::Session,
    mut stream: actix_ws::MessageStream,
    data: data_engine::Proxy,
    run_id: Uuid,
    secret: String,
) {
    let Some(claims) = handshake(&mut session, &mut stream, &secret).await else {
        return;
    };
    let caller = auth::Caller::User(claims.sub);
    match data
        .events(&caller, &format!("/runs/{run_id}/events"))
        .await
    {
        Ok(frames) => relay(&mut session, frames).await,
        Err(error) => {
            let _ = session.text(ws::error_frame(&error.to_string())).await;
        }
    }
    let _ = session.close(None).await;
}

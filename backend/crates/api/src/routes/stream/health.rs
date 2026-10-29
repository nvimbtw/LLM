//! Administrator subscriptions to the backend's current service observations.

use std::time::Duration;

use futures_util::StreamExt;
use service::health::Status;
use tokio::sync::watch;

use super::connection::handshake;
use crate::ws;

/// Authenticate an administrator and send bounded full snapshots until disconnect or shutdown.
pub(super) async fn run(
    mut session: actix_ws::Session,
    mut stream: actix_ws::MessageStream,
    mut health: watch::Receiver<Vec<Status>>,
    secret: String,
) {
    let Ok(Some(claims)) = tokio::time::timeout(
        Duration::from_secs(10),
        handshake(&mut session, &mut stream, &secret),
    )
    .await
    else {
        return;
    };
    if !matches!(claims.access_level, auth::AccessLevel::Admin) {
        let _ = session.text(ws::error_frame("admin only")).await;
        let _ = session.close(None).await;
        return;
    }
    let mut refresh = tokio::time::interval(Duration::from_secs(15));
    loop {
        tokio::select! {
            changed = health.changed() => if changed.is_err() { break; },
            _ = refresh.tick() => {},
            message = stream.next() => match message {
                Some(Ok(actix_ws::Message::Ping(bytes))) => { if session.pong(&bytes).await.is_err() { break; } continue; },
                Some(Ok(actix_ws::Message::Close(_))) | None | Some(Err(_)) => break,
                _ => continue,
            },
        }
        let snapshot =
            serde_json::json!({"type": "services", "services": &*health.borrow_and_update()})
                .to_string();
        if !matches!(
            tokio::time::timeout(Duration::from_secs(5), session.text(snapshot)).await,
            Ok(Ok(()))
        ) {
            break;
        }
    }
    let _ = session.close(None).await;
}

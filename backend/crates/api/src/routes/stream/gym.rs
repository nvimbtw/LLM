//! Admin socket composition for gym runs.

use super::connection::{handshake, next_json};
use crate::{gym, ws};
use llm_engine::Proxy;
use uuid::Uuid;

/// Authenticate an admin and pass optional sampling settings to the gym driver; non-admins receive an error.
pub(super) async fn run(
    mut session: actix_ws::Session,
    mut stream: actix_ws::MessageStream,
    engine: Proxy,
    gyms: gyms_engine::Proxy,
    run_id: Uuid,
    secret: String,
) {
    let Some(claims) = handshake(&mut session, &mut stream, &secret).await else {
        return;
    };
    if !matches!(claims.access_level, auth::AccessLevel::Admin) {
        let _ = session.text(ws::error_frame("admin only")).await;
        let _ = session.close(None).await;
        return;
    }
    let Some(ask) = next_json(&mut stream).await else {
        return;
    };
    gym::run(
        &mut session,
        &engine,
        &gyms,
        &auth::Caller::User(claims.sub),
        run_id,
        &ask,
    )
    .await;
    let _ = session.close(None).await;
}

//! A stored chat turn: prepare, stream, join tools, then continue with authorized history.

mod relay;
mod request;
mod tools;

use ::tools::Registry;
use llm_engine::Proxy;
use serde_json::{json, Value};
use tokio::sync::mpsc;

pub use crate::streaming::{envelope, string_at};
use crate::{conversation, ws};
pub use request::carry_settings;
use request::{drain_interjection, fold_interjection, max_tool_rounds};

pub struct ChatTurn<'a> {
    engine: &'a Proxy,
    registry: &'a Registry,
    pool: &'a sqlx::PgPool,
    owner: uuid::Uuid,
    visibility: &'static str,
    chat_id: uuid::Uuid,
    tool_session: String,
}

/// Drive one chat turn with its authenticated resources; report failures on the socket.
/// The conversation lock spans all rounds and their persistence; the chat UUID also owns its tool session.
pub async fn run(
    session: &mut actix_ws::Session,
    turn: ChatTurn<'_>,
    ask: Value,
    interjections: mpsc::UnboundedReceiver<String>,
) {
    if let Err(error) = turn.drive(session, ask, interjections).await {
        let _ = session.text(ws::error_frame(&error.to_string())).await;
    }
}

impl<'a> ChatTurn<'a> {
    /// Bind service/storage dependencies to one authenticated conversation owner and derive its stable tool-session key.
    pub fn new(
        engine: &'a Proxy,
        registry: &'a Registry,
        pool: &'a sqlx::PgPool,
        owner: uuid::Uuid,
        visibility: &'static str,
        chat_id: uuid::Uuid,
    ) -> Self {
        Self {
            engine,
            registry,
            pool,
            owner,
            visibility,
            chat_id,
            tool_session: chat_id.to_string(),
        }
    }

    /// Run bounded rounds under one conversation lease, stopping on errors, disconnect or a reply without calls.
    async fn drive(
        &self,
        session: &mut actix_ws::Session,
        mut ask: Value,
        mut interjections: mpsc::UnboundedReceiver<String>,
    ) -> Result<(), service::Error> {
        let _guard = conversation::lock(self.chat_id).await;
        let caller = auth::Caller::User(self.owner);
        let settings = ask.clone();
        let rounds = max_tool_rounds();
        for round in 0..rounds {
            let prepared = conversation::prepare(
                self.pool,
                self.engine,
                self.owner,
                self.visibility,
                self.chat_id,
                ask,
            )
            .await?;
            let frames = self
                .engine
                .stream(&caller, "/chat/stream", prepared.config.clone())
                .await?;
            let Some(pending) =
                relay::relay(session, self, frames, round + 1 == rounds, &prepared).await
            else {
                return Ok(());
            };
            if pending.is_empty() {
                return Ok(());
            }
            let Some(mut next) = tools::answer(session, pending).await else {
                return Ok(());
            };
            carry_settings(&settings, &mut next);
            if !confirm_interjection(session, &mut next, &mut interjections).await {
                return Ok(());
            }
            ask = next;
        }
        Ok(())
    }
}

/// Fold buffered steering into the next request and acknowledge it before the next generation starts.
async fn confirm_interjection(
    session: &mut actix_ws::Session,
    next: &mut Value,
    receiver: &mut mpsc::UnboundedReceiver<String>,
) -> bool {
    let Some(text) = fold_interjection(next, drain_interjection(receiver)) else {
        return true;
    };
    session
        .text(envelope("interjected", &json!({"text":text})).to_string())
        .await
        .is_ok()
}

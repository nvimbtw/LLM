//! One connection owns one question, its reader and all chat tool rounds.

use super::{
    connection::{handshake, next_json},
    reader::{self, SocketReader},
};
use crate::turn::{self, ChatTurn};
use llm_engine::Proxy;
use tools::Registry;
use uuid::Uuid;

pub(super) struct Connection {
    pub engine: Proxy,
    pub registry: Registry,
    pub pool: sqlx::PgPool,
    pub chat_id: Uuid,
    pub secret: String,
}

/// Authenticate, accept one question, then own a concurrent steering reader until the turn ends.
pub(super) async fn run(
    mut session: actix_ws::Session,
    mut stream: actix_ws::MessageStream,
    connection: Connection,
) {
    let Some(claims) = handshake(&mut session, &mut stream, &connection.secret).await else {
        return;
    };
    let Some(ask) = next_json(&mut stream).await else {
        return;
    };
    let (sender, interjections) = tokio::sync::mpsc::unbounded_channel();
    let reader = SocketReader::new(actix_web::rt::spawn(reader::interjections(stream, sender)));
    let turn = ChatTurn::new(
        &connection.engine,
        &connection.registry,
        &connection.pool,
        claims.sub,
        connection.chat_id,
    );
    turn::run(&mut session, turn, ask, interjections).await;
    drop(reader);
    let _ = session.close(None).await;
}

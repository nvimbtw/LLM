//! Exercise administrator admission and snapshot delivery over a real browser socket.

use actix_web::{web, App};
use auth::{token, AccessLevel};
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use service::health::{State, Status};
use tokio::sync::watch;
use uuid::Uuid;

const SECRET: &str = "service-health-test";

/// Produce the encrypted user credential expected by browser sockets.
fn credential(level: AccessLevel) -> String {
    let token = token::issue_user_token(Uuid::new_v4(), level, SECRET).unwrap();
    auth::token_cipher::encrypt(&token, SECRET).unwrap()
}

macro_rules! frame {
    ($socket:expr) => {{
        let awc::ws::Frame::Text(text) = $socket.next().await.unwrap().unwrap() else {
            panic!("expected text frame")
        };
        serde_json::from_slice::<serde_json::Value>(&text).unwrap()
    }};
}

/// Only admins receive initial and changed service snapshots; invalid credentials are rejected.
#[actix_web::test]
async fn admin_subscription_is_authenticated_and_live() {
    let (sender, receiver) = watch::channel(vec![Status {
        id: "llm-engine".into(),
        state: State::Reconnecting,
        detail: None,
    }]);
    let server = actix_test::start(move || {
        App::new()
            .app_data(web::Data::new(SECRET.to_owned()))
            .app_data(web::Data::new(receiver.clone()))
            .configure(api::configure)
    });
    for (token, admin) in [
        (credential(AccessLevel::User), false),
        (credential(AccessLevel::Admin), true),
    ] {
        let (_, mut socket) = awc::Client::new()
            .ws(server.url("/ws/services"))
            .connect()
            .await
            .unwrap();
        socket
            .send(awc::ws::Message::Text(
                json!({"token": token}).to_string().into(),
            ))
            .await
            .unwrap();
        assert_eq!(frame!(socket)["type"], "ready");
        let first = frame!(socket);
        if !admin {
            assert_eq!(first["error"], "admin only");
            continue;
        }
        assert_eq!(first["services"][0]["state"], "reconnecting");
        sender.send_modify(|statuses| statuses[0].state = State::Online);
        assert_eq!(frame!(socket)["services"][0]["state"], "online");
    }
    let (_, mut socket) = awc::Client::new()
        .ws(server.url("/ws/services"))
        .connect()
        .await
        .unwrap();
    socket
        .send(awc::ws::Message::Text(
            json!({"token": "invalid"}).to_string().into(),
        ))
        .await
        .unwrap();
    assert_eq!(frame!(socket)["error"], "authentication failed");
}

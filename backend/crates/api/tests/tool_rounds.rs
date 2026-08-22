//! A turn that calls a tool, over the socket a browser actually opens.
//!
//! The engine is a stub emitting a fixed SSE script. What is under test is this
//! side: that a `tool_call` starts work, that the turn comes back for a second
//! round carrying the result, and that it stops rather than looping.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc::Receiver;

use actix_web::{web, App};
use auth::token;
use auth::AccessLevel;
use engine::Proxy;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tools::Registry;
use uuid::Uuid;

const JWT_SECRET: &str = "tool-rounds-test-jwt-secret";
const SERVICE_SECRET: &str = "tool-rounds-test-service-secret";

/// A round the stub engine plays: SSE frames, in order.
const CALLS_A_TOOL: &str = concat!(
    "event: token\ndata: {\"token\": \"<|tool-call|>\"}\n\n",
    "event: token\ndata: {\"token\": \"{\\\"tool\\\": \\\"web-search\\\", \\\"argument\\\": \\\"elbrus height\\\"}\"}\n\n",
    "event: tool_call\ndata: {\"tool\": \"web-search\", \"argument\": \"elbrus height\"}\n\n",
    "event: done\ndata: {\"message_id\": \"m1\", \"content\": \"asked\"}\n\n",
);

const ANSWERS_PLAINLY: &str = concat!(
    "event: token\ndata: {\"token\": \"5,642 m\"}\n\n",
    "event: done\ndata: {\"message_id\": \"m2\", \"content\": \"answered\"}\n\n",
);

/// An engine that plays `script[n]` to the nth request and records each body.
///
/// Requests past the end get the last script again, so a test that expects the
/// loop to stop fails by hanging on its own assertion rather than by the stub
/// running out.
fn stub(script: Vec<&'static str>) -> (String, Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("addr"));
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        for (nth, socket) in listener.incoming().enumerate() {
            let Ok(mut socket) = socket else { return };
            let mut buffer = [0u8; 8192];
            let read = socket.read(&mut buffer).unwrap_or(0);

            let body = script[nth.min(script.len() - 1)];
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = socket.write_all(response.as_bytes());
            let _ = socket.flush();
            if tx.send(String::from_utf8_lossy(&buffer[..read]).to_string()).is_err() {
                return;
            }
        }
    });

    (url, rx)
}

fn server(url: String, registry: Registry) -> actix_test::TestServer {
    actix_test::start(move || {
        App::new()
            .app_data(web::Data::new(JWT_SECRET.to_string()))
            .app_data(web::Data::new(Proxy::new(
                url.clone(),
                Some(SERVICE_SECRET.to_string()),
            )))
            .app_data(web::Data::new(registry.clone()))
            .configure(api::configure)
    })
}

fn sealed_token() -> String {
    let jwt = token::issue_user_token(Uuid::new_v4(), AccessLevel::User, JWT_SECRET)
        .expect("issued");
    auth::token_cipher::encrypt(&jwt, JWT_SECRET).expect("sealed")
}

/// Ask a chat one question and collect every frame until the socket closes.
async fn converse(server: &mut actix_test::TestServer) -> Vec<Value> {
    let mut socket = server
        .ws_at(&format!("/ws/chats/{}", Uuid::new_v4()))
        .await
        .expect("socket opened");

    socket
        .send(actix_http::ws::Message::Text(
            json!({ "token": sealed_token() }).to_string().into(),
        ))
        .await
        .expect("auth frame sent");

    let mut frames = Vec::new();
    let mut asked = false;
    while let Some(Ok(message)) = socket.next().await {
        let actix_http::ws::Frame::Text(bytes) = message else { continue };
        let frame: Value = serde_json::from_slice(&bytes).expect("json frame");

        if frame["type"] == "ready" && !asked {
            asked = true;
            socket
                .send(actix_http::ws::Message::Text(
                    json!({ "model_name": "tiny", "text": "how tall is elbrus?"})
                        .to_string()
                        .into(),
                ))
                .await
                .expect("question sent");
            continue;
        }
        frames.push(frame);
    }
    frames
}

fn kinds(frames: &[Value]) -> Vec<&str> {
    frames.iter().filter_map(|f| f["type"].as_str()).collect()
}

fn body_of(raw: &str) -> Value {
    let body = raw.split("\r\n\r\n").nth(1).unwrap_or_default();
    serde_json::from_str(body.trim_end_matches('\0')).unwrap_or(Value::Null)
}

#[actix_web::test]
async fn a_reply_that_calls_a_tool_comes_back_for_a_second_round() {
    let (url, asked) = stub(vec![CALLS_A_TOOL, ANSWERS_PLAINLY]);
    let mut server = server(url, Registry::new(vec![]));

    let frames = converse(&mut server).await;

    // The call is forwarded as it happens, the result once it is in, and the
    // second round's tokens follow both.
    assert_eq!(
        kinds(&frames),
        vec!["token", "token", "tool_call", "done", "tool_result", "token", "done"]
    );

    // Only the last `done` ends the turn. The client closes on that flag, so a
    // first round marked final would cut the answer off before it arrived.
    let dones: Vec<&Value> = frames.iter().filter(|f| f["type"] == "done").collect();
    assert_eq!(dones[0]["final"], false);
    assert_eq!(dones[1]["final"], true);

    let first = body_of(&asked.recv().expect("a first request"));
    assert_eq!(first["text"], "how tall is elbrus?");
    assert!(first.get("tool_result").is_none());

    let second = body_of(&asked.recv().expect("a second request"));
    assert_eq!(second["tool_result"]["tool"], "web-search");
    assert_eq!(second["tool_result"]["argument"], "elbrus height");
    // Carried across the round; the question is not, because the engine takes
    // exactly one of `text` and `tool_result`.
    assert_eq!(second["model_name"], "tiny");
    assert!(second.get("text").is_none());
}

#[actix_web::test]
async fn a_tool_with_no_credentials_answers_with_a_caveat_rather_than_nothing() {
    let (url, asked) = stub(vec![CALLS_A_TOOL, ANSWERS_PLAINLY]);
    let mut server = server(url, Registry::new(vec![]));

    let frames = converse(&mut server).await;

    let result = frames.iter().find(|f| f["type"] == "tool_result").expect("a result frame");
    assert!(result["body"].as_str().expect("a body").starts_with("<|warn|>"));

    let _ = asked.recv();
    let second = body_of(&asked.recv().expect("a second request"));
    assert!(second["tool_result"]["body"].as_str().expect("a body").contains("<|warn|>"));
}

#[actix_web::test]
async fn a_reply_that_asks_for_nothing_is_one_round() {
    let (url, asked) = stub(vec![ANSWERS_PLAINLY]);
    let mut server = server(url, Registry::new(vec![]));

    let frames = converse(&mut server).await;
    assert_eq!(kinds(&frames), vec!["token", "done"]);
    assert_eq!(frames[1]["final"], true);

    assert!(asked.recv().is_ok());
    assert!(asked.recv_timeout(std::time::Duration::from_millis(200)).is_err());
}

#[actix_web::test]
async fn a_model_that_calls_a_tool_every_round_is_stopped_rather_than_looping() {
    let (url, asked) = stub(vec![CALLS_A_TOOL]);
    let mut server = server(url, Registry::new(vec![]));

    let frames = converse(&mut server).await;

    // The round that hits the ceiling is final even though it called a tool,
    // so the client stops waiting for an answer that is never coming.
    let dones: Vec<&Value> = frames.iter().filter(|f| f["type"] == "done").collect();
    assert_eq!(dones.last().expect("a done frame")["final"], true);

    // Two rounds, and then the turn ends despite the engine still asking.
    assert!(asked.recv().is_ok());
    assert!(asked.recv().is_ok());
    assert!(asked.recv_timeout(std::time::Duration::from_millis(200)).is_err());
}

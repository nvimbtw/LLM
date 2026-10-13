//! A turn that calls a tool, over the socket a browser actually opens.
//!
//! The LLM engine is a stub emitting a fixed SSE script. What is under test is this
//! side: that a `tool_call` starts work, that the turn comes back for a second
//! round carrying the result, and that it stops rather than looping.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc::Receiver;

use actix_web::{web, App};
use auth::token;
use auth::AccessLevel;
use futures_util::{SinkExt, StreamExt};
use llm_engine::Proxy;
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
    "event: done\ndata: {\"text\": \"asked\", \"generated_tokens\": 4}\n\n",
);

const ANSWERS_PLAINLY: &str = concat!(
    "event: token\ndata: {\"token\": \"5,642 m\"}\n\n",
    "event: done\ndata: {\"text\": \"answered\", \"generated_tokens\": 4}\n\n",
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
        let mut nth = 0;
        for socket in listener.incoming() {
            let Ok(mut socket) = socket else { return };
            let mut buffer = [0u8; 8192];
            let read = socket.read(&mut buffer).unwrap_or(0);

            let request = String::from_utf8_lossy(&buffer[..read]).to_string();
            if request.starts_with("POST /chat/prepare ") {
                let ask = body_of(&request);
                let turns = if let Some(text) = ask["text"].as_str() {
                    json!([{"role":"user","text":text,"tokens":4}])
                } else {
                    json!([{"role":"tool","text":ask["tool_result"]["body"],"tokens":5}])
                };
                let body = json!({"turns":turns,"model_name":"tiny","model_id":null}).to_string();
                let response = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len());
                let _ = socket.write_all(response.as_bytes());
                let _ = tx.send(request);
                continue;
            }
            let body = script[nth.min(script.len() - 1)];
            nth += 1;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = socket.write_all(response.as_bytes());
            let _ = socket.flush();
        }
    });

    (url, rx)
}

fn server(url: String, registry: Registry, pool: sqlx::PgPool) -> actix_test::TestServer {
    actix_test::start(move || {
        App::new()
            .app_data(web::Data::new(JWT_SECRET.to_string()))
            .app_data(web::Data::new(Proxy::new(
                url.clone(),
                Some(SERVICE_SECRET.to_string()),
            )))
            .app_data(web::Data::new(registry.clone()))
            .app_data(web::Data::new(pool.clone()))
            .configure(api::configure)
    })
}

fn sealed_token(owner: Uuid) -> String {
    let jwt = token::issue_user_token(owner, AccessLevel::User, JWT_SECRET).expect("issued");
    auth::token_cipher::encrypt(&jwt, JWT_SECRET).expect("sealed")
}

/// Ask a chat one question and collect every frame until the socket closes.
async fn converse(server: &mut actix_test::TestServer, pool: &sqlx::PgPool) -> Vec<Value> {
    let owner = db::user::create(pool, &Uuid::new_v4().to_string(), "hash", None)
        .await
        .unwrap()
        .id_user;
    let chat = db::conversation::create(pool, owner, "test").await.unwrap();
    let mut socket = server
        .ws_at(&format!("/ws/chats/{}", chat.id))
        .await
        .expect("socket opened");

    socket
        .send(actix_http::ws::Message::Text(
            json!({ "token": sealed_token(owner) }).to_string().into(),
        ))
        .await
        .expect("auth frame sent");

    let mut frames = Vec::new();
    let mut asked = false;
    while let Some(Ok(message)) = socket.next().await {
        let actix_http::ws::Frame::Text(bytes) = message else {
            continue;
        };
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

#[sqlx::test(migrations = "../db/migrations")]
async fn a_reply_that_calls_a_tool_comes_back_for_a_second_round(pool: sqlx::PgPool) {
    let (url, asked) = stub(vec![CALLS_A_TOOL, ANSWERS_PLAINLY]);
    let mut server = server(url, Registry::new(vec![]), pool.clone());

    let frames = converse(&mut server, &pool).await;

    // The call is forwarded as it happens, the result once it is in, and the
    // second round's tokens follow both.
    assert_eq!(
        kinds(&frames),
        vec![
            "token",
            "token",
            "tool_call",
            "done",
            "tool_result",
            "token",
            "done"
        ]
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
    // Carried across the round; the question is not, because the LLM engine takes
    // exactly one of `text` and `tool_result`.
    assert_eq!(second["model_name"], "tiny");
    assert!(second.get("text").is_none());
}

#[sqlx::test(migrations = "../db/migrations")]
async fn a_tool_with_no_credentials_answers_with_a_caveat_rather_than_nothing(pool: sqlx::PgPool) {
    let (url, asked) = stub(vec![CALLS_A_TOOL, ANSWERS_PLAINLY]);
    let mut server = server(url, Registry::new(vec![]), pool.clone());

    let frames = converse(&mut server, &pool).await;

    let result = frames
        .iter()
        .find(|f| f["type"] == "tool_result")
        .expect("a result frame");
    assert!(result["body"]
        .as_str()
        .expect("a body")
        .starts_with("<|warn|>"));

    let _ = asked.recv();
    let second = body_of(&asked.recv().expect("a second request"));
    assert!(second["tool_result"]["body"]
        .as_str()
        .expect("a body")
        .contains("<|warn|>"));
}

#[sqlx::test(migrations = "../db/migrations")]
async fn a_reply_that_asks_for_nothing_is_one_round(pool: sqlx::PgPool) {
    let (url, asked) = stub(vec![ANSWERS_PLAINLY]);
    let mut server = server(url, Registry::new(vec![]), pool.clone());

    let frames = converse(&mut server, &pool).await;
    assert_eq!(kinds(&frames), vec!["token", "done"]);
    assert_eq!(frames[1]["final"], true);

    assert!(asked.recv().is_ok());
    assert!(asked
        .recv_timeout(std::time::Duration::from_millis(200))
        .is_err());
}

#[sqlx::test(migrations = "../db/migrations")]
async fn a_model_that_calls_a_tool_every_round_is_stopped_rather_than_looping(pool: sqlx::PgPool) {
    let (url, asked) = stub(vec![CALLS_A_TOOL]);
    let mut server = server(url, Registry::new(vec![]), pool.clone());

    let frames = converse(&mut server, &pool).await;

    // The round that hits the ceiling is final even though it called a tool,
    // so the client stops waiting for an answer that is never coming.
    let dones: Vec<&Value> = frames.iter().filter(|f| f["type"] == "done").collect();
    assert_eq!(dones.last().expect("a done frame")["final"], true);

    // The default permits six rounds before ending a turn that keeps calling.
    for _ in 0..6 {
        assert!(asked.recv().is_ok());
    }
    assert!(asked
        .recv_timeout(std::time::Duration::from_millis(200))
        .is_err());
}

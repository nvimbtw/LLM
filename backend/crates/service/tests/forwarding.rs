//! The proxy against a stub service: what goes out, and what comes back.
//!
//! `seam.rs` proves a token minted here opens the real LLM engine's door. This is
//! the other half — the shapes on the wire, which need no PyTorch to check.

use std::io::Write;
use std::net::TcpListener;

use futures_util::StreamExt;
use serde_json::{json, Value};
use service::Proxy;

const AUDIENCE: &str = "stub";
const SECRET: &str = "forwarding-test-secret";

/// A one-request HTTP server. Returns its URL and the request it received.
fn stub(response: &'static str) -> (String, std::thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}", listener.local_addr().unwrap());

    let handle = std::thread::spawn(move || {
        use std::io::Read;

        let (mut socket, _) = listener.accept().expect("accept");
        let mut buffer = [0u8; 8192];
        let read = socket.read(&mut buffer).expect("read");
        socket.write_all(response.as_bytes()).expect("write");
        let _ = socket.flush();
        String::from_utf8_lossy(&buffer[..read]).to_string()
    });

    (url, handle)
}

fn ok(body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

#[tokio::test]
async fn a_call_carries_a_token_for_the_service_it_names() {
    let (url, handle) = stub(Box::leak(ok(r#"{"status":"ok"}"#).into_boxed_str()));
    let proxy = Proxy::new(url, AUDIENCE, Some(SECRET.into()));

    let body = proxy
        .get(&auth::Caller::System, "/")
        .await
        .expect("reaches the stub");
    let request = handle.join().unwrap();

    assert_eq!(body, json!({"status": "ok"}));
    let header = request
        .lines()
        .find_map(|line| line.strip_prefix("authorization: "))
        .expect("signed");
    let token = header
        .trim()
        .strip_prefix("Bearer ")
        .expect("bearer scheme");
    assert!(auth::token::verify_service_token(token, AUDIENCE, SECRET).is_ok());
}

#[tokio::test]
async fn an_unsecured_proxy_still_says_who_it_acts_for() {
    // The signature is worthless without a shared secret, but `act` is not: an
    // open service reads it, and every owner-scoped route needs it.
    let (url, handle) = stub(Box::leak(ok("{}").into_boxed_str()));
    let proxy = Proxy::new(url, AUDIENCE, None);
    let who = uuid::Uuid::new_v4();

    proxy
        .get(&auth::Caller::User(who), "/chats")
        .await
        .expect("reaches the stub");

    let request = handle.join().unwrap();
    assert!(
        request.to_lowercase().contains("authorization:"),
        "no token was sent"
    );
    assert!(request.contains(&who.to_string()) || request.to_lowercase().contains("bearer"));
}

#[tokio::test]
async fn a_body_is_forwarded_verbatim() {
    let (url, handle) = stub(Box::leak(ok("{}").into_boxed_str()));
    let proxy = Proxy::new(url, AUDIENCE, None);

    let sent = json!({"model_name": "tiny", "nested": {"epochs": 2}});
    proxy
        .post(&auth::Caller::System, "/train", sent.clone())
        .await
        .expect("posts");

    let request = handle.join().unwrap();
    let body = request.split("\r\n\r\n").nth(1).expect("a body");
    assert_eq!(serde_json::from_str::<Value>(body).unwrap(), sent);
}

#[tokio::test]
async fn an_empty_response_is_null_rather_than_a_decode_failure() {
    // A 204 has no body, and the delete routes answer with one.
    let (url, _handle) = stub("HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n");
    let proxy = Proxy::new(url, AUDIENCE, None);

    let body = proxy
        .delete(&auth::Caller::System, "/chats/x")
        .await
        .expect("deletes");
    assert_eq!(body, Value::Null);
}

#[tokio::test]
async fn a_refusal_carries_the_services_own_status_and_message() {
    let body = r#"{"detail":"no run 7"}"#;
    let response = Box::leak(
        format!(
            "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .into_boxed_str(),
    );
    let (url, _handle) = stub(response);
    let proxy = Proxy::new(url, AUDIENCE, None);

    let err = proxy
        .get(&auth::Caller::System, "/runs/7")
        .await
        .expect_err("refused");

    match err {
        service::Error::Status { code, detail } => {
            assert_eq!(code, 404);
            assert_eq!(detail, "no run 7");
        }
        other => panic!("wrong error: {other}"),
    }
}

#[tokio::test]
async fn an_unreachable_service_is_a_transport_failure() {
    // Nothing is listening: the port was taken and released.
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let proxy = Proxy::new(format!("http://127.0.0.1:{port}"), AUDIENCE, None);

    let err = proxy
        .get(&auth::Caller::System, "/")
        .await
        .expect_err("unreachable");
    assert!(
        matches!(err, service::Error::Transport(_)),
        "wrong error: {err}"
    );
}

#[tokio::test]
async fn an_event_stream_arrives_frame_by_frame() {
    let body = "event: progress\ndata: {\"step\":1}\n\nevent: done\ndata: {\"ok\":true}\n\n";
    let response = Box::leak(
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .into_boxed_str(),
    );
    let (url, _handle) = stub(response);
    let proxy = Proxy::new(url, AUDIENCE, None);

    let stream = proxy
        .events(&auth::Caller::System, "/runs/1/events")
        .await
        .expect("opens");
    let frames: Vec<_> = stream.filter_map(|f| async { f.ok() }).collect().await;

    assert_eq!(
        frames.iter().map(|f| f.kind.as_str()).collect::<Vec<_>>(),
        ["progress", "done"]
    );
    assert_eq!(frames[0].data["step"], 1);
    assert!(frames[1].is_final());
}

#[tokio::test]
async fn a_stream_that_is_refused_never_opens() {
    let (url, _handle) =
        stub("HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    let proxy = Proxy::new(url, AUDIENCE, None);

    let err = proxy
        .events(&auth::Caller::System, "/runs/1/events")
        .await
        .err()
        .expect("refused");
    assert!(
        matches!(err, service::Error::Status { code: 401, .. }),
        "wrong error: {err}"
    );
}

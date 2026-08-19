//! The proxy routes end to end: who is let through, and what is forwarded.
//!
//! The engine is a stub — what is under test is this side. The engine's own
//! behaviour is covered by its suite, and the hop between the two by
//! `engine/tests/seam.rs`.

use std::io::{Read, Write};
use std::net::TcpListener;

use actix_web::{App, http::StatusCode, test, web};
use auth::AccessLevel;
use auth::token::{self, Caller};
use engine::Proxy;
use serde_json::{Value, json};
use uuid::Uuid;

const JWT_SECRET: &str = "proxying-test-jwt-secret";
const SERVICE_SECRET: &str = "proxying-test-service-secret";

/// A stub engine that answers `body` to everything, recording what it was asked.
fn stub(body: &'static str) -> (String, std::sync::mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        for socket in listener.incoming() {
            let Ok(mut socket) = socket else { return };
            let mut buffer = [0u8; 8192];
            let read = socket.read(&mut buffer).unwrap_or(0);
            let _ = socket.write_all(response.as_bytes());
            let _ = socket.flush();
            if tx.send(String::from_utf8_lossy(&buffer[..read]).to_string()).is_err() {
                return;
            }
        }
    });

    (url, rx)
}

/// The encrypted bearer a browser holds, for a person at `level`.
fn session(level: AccessLevel) -> (Uuid, String) {
    let who = Uuid::new_v4();
    let jwt = token::issue_user_token(who, level, JWT_SECRET).expect("issued");
    let sealed = auth::token_cipher::encrypt(&jwt, JWT_SECRET).expect("sealed");
    (who, format!("Bearer {sealed}"))
}

macro_rules! service {
    ($url:expr) => {
        test::init_service(
            App::new()
                .app_data(web::Data::new(JWT_SECRET.to_string()))
                .app_data(web::Data::new(Proxy::new(
                    $url,
                    Some(SERVICE_SECRET.to_string()),
                )))
                .app_data(web::Data::new(tools::Registry::new(vec![])))
                .configure(api::configure),
        )
        .await
    };
}

fn request_line(raw: &str) -> String {
    raw.lines().next().unwrap_or_default().to_string()
}

fn acting_for(raw: &str) -> Caller {
    let header = raw
        .lines()
        .find_map(|line| line.strip_prefix("authorization: "))
        .expect("the hop is signed");
    let service = header.trim().strip_prefix("Bearer ").expect("bearer scheme");
    token::verify_service_token(service, engine::AUDIENCE, SERVICE_SECRET).expect("valid")
}

#[actix_web::test]
async fn an_unauthenticated_request_never_reaches_the_engine() {
    let (url, asked) = stub("{}");
    let app = service!(url);

    let response = test::call_service(&app, test::TestRequest::get().uri("/chats").to_request()).await;

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert!(asked.try_recv().is_err(), "the engine was called anyway");
}

#[actix_web::test]
async fn a_signed_in_person_is_forwarded_as_themselves() {
    let (url, asked) = stub(r#"{"chats":[]}"#);
    let app = service!(url);
    let (who, bearer) = session(AccessLevel::User);

    let response = test::call_service(
        &app,
        test::TestRequest::get()
            .uri("/chats")
            .insert_header(("authorization", bearer))
            .to_request(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let raw = asked.recv().expect("the engine was called");
    assert_eq!(request_line(&raw), "GET /chats HTTP/1.1");
    assert!(matches!(acting_for(&raw), Caller::User(id) if id == who));
}

#[actix_web::test]
async fn the_engines_answer_is_passed_through_untouched() {
    let (url, _asked) = stub(r#"{"chats":[{"chat_id":"7","title":"hi","extra":true}]}"#);
    let app = service!(url);
    let (_, bearer) = session(AccessLevel::User);

    let response = test::call_service(
        &app,
        test::TestRequest::get()
            .uri("/chats")
            .insert_header(("authorization", bearer))
            .to_request(),
    )
    .await;

    let body: Value = test::read_body_json(response).await;
    assert_eq!(body["chats"][0]["extra"], json!(true));
}

#[actix_web::test]
async fn starting_a_run_is_admin_work() {
    let (url, asked) = stub("{}");
    let app = service!(url);
    let (_, bearer) = session(AccessLevel::User);

    let response = test::call_service(
        &app,
        test::TestRequest::post()
            .uri("/runs/train")
            .insert_header(("authorization", bearer))
            .set_json(json!({"model_name": "x"}))
            .to_request(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    assert!(asked.try_recv().is_err(), "the engine was called anyway");
}

#[actix_web::test]
async fn watching_a_run_is_not() {
    let (url, asked) = stub(r#"{"active":[],"history":[]}"#);
    let app = service!(url);
    let (_, bearer) = session(AccessLevel::User);

    let response = test::call_service(
        &app,
        test::TestRequest::get()
            .uri("/runs")
            .insert_header(("authorization", bearer))
            .to_request(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(request_line(&asked.recv().unwrap()), "GET /runs HTTP/1.1");
}

#[actix_web::test]
async fn a_query_string_is_forwarded_rather_than_restated() {
    let (url, asked) = stub(r#"{"active":[],"history":[]}"#);
    let app = service!(url);
    let (_, bearer) = session(AccessLevel::User);

    test::call_service(
        &app,
        test::TestRequest::get()
            .uri("/runs?limit=2&offset=4")
            .insert_header(("authorization", bearer))
            .to_request(),
    )
    .await;

    assert_eq!(request_line(&asked.recv().unwrap()), "GET /runs?limit=2&offset=4 HTTP/1.1");
}

#[actix_web::test]
async fn an_admin_action_reaches_the_engine_as_the_backend_itself() {
    // Deleting a model is not aimed at anyone's data, so it acts as the system.
    let (url, asked) = stub("{}");
    let app = service!(url);
    let (_, bearer) = session(AccessLevel::Admin);
    let model = Uuid::new_v4();

    let response = test::call_service(
        &app,
        test::TestRequest::delete()
            .uri(&format!("/models/{model}"))
            .insert_header(("authorization", bearer))
            .to_request(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let raw = asked.recv().unwrap();
    assert_eq!(request_line(&raw), format!("DELETE /models/{model} HTTP/1.1"));
    assert!(matches!(acting_for(&raw), Caller::System));
}

#[actix_web::test]
async fn a_run_the_engine_refuses_is_refused_here_with_the_same_status() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for socket in listener.incoming() {
            let Ok(mut socket) = socket else { return };
            let mut buffer = [0u8; 4096];
            let _ = socket.read(&mut buffer);
            let body = r#"{"detail":"no run 7"}"#;
            let _ = socket.write_all(
                format!(
                    "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            );
        }
    });

    let app = service!(url);
    let (_, bearer) = session(AccessLevel::User);

    let response = test::call_service(
        &app,
        test::TestRequest::get()
            .uri(&format!("/runs/{}", Uuid::new_v4()))
            .insert_header(("authorization", bearer))
            .to_request(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body: Value = test::read_body_json(response).await;
    assert!(body["error"].as_str().unwrap().contains("no run 7"));
}

#[actix_web::test]
async fn an_unreachable_engine_is_a_bad_gateway_rather_than_a_crash() {
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let app = service!(format!("http://127.0.0.1:{port}"));
    let (_, bearer) = session(AccessLevel::User);

    let response = test::call_service(
        &app,
        test::TestRequest::get()
            .uri("/chats")
            .insert_header(("authorization", bearer))
            .to_request(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
}

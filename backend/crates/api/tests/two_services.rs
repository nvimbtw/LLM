//! Two services behind one server.
//!
//! Actix keys `app_data` by type, so registering both proxies as the same type
//! makes one of them silently win and every route reach it. That failure is
//! invisible in a build and in most tests — a `/data` route would simply answer
//! with whatever the LLM engine said. The newtype in each service crate is what
//! prevents it, and this is the test that would notice if it were removed.

use std::io::{Read, Write};
use std::net::TcpListener;

use actix_web::{App, http::StatusCode, test, web};
use auth::AccessLevel;
use auth::token;
use uuid::Uuid;

const JWT_SECRET: &str = "two-services-test-jwt-secret";

/// A stub that names itself in every answer, so a reply identifies its sender.
fn stub(name: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}", listener.local_addr().unwrap());
    let body = format!("{{\"answered_by\":\"{name}\"}}");

    std::thread::spawn(move || {
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        for socket in listener.incoming() {
            let Ok(mut socket) = socket else { return };
            let mut buffer = [0u8; 8192];
            let _ = socket.read(&mut buffer);
            let _ = socket.write_all(response.as_bytes());
            let _ = socket.flush();
        }
    });

    url
}

fn session(level: AccessLevel) -> String {
    let jwt = token::issue_user_token(Uuid::new_v4(), level, JWT_SECRET).expect("issued");
    let sealed = auth::token_cipher::encrypt(&jwt, JWT_SECRET).expect("sealed");
    format!("Bearer {sealed}")
}

/// A stub that hands back the `Authorization` header it was given, so a test can
/// read the token the backend minted on its way out.
fn echoing() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}", listener.local_addr().unwrap());

    std::thread::spawn(move || {
        for socket in listener.incoming() {
            let Ok(mut socket) = socket else { return };
            let mut buffer = [0u8; 8192];
            let read = socket.read(&mut buffer).unwrap_or(0);
            let request = String::from_utf8_lossy(&buffer[..read]).to_string();
            let sent = request
                .lines()
                .find_map(|line| line.to_ascii_lowercase().starts_with("authorization:").then(
                    || line.split_once(':').map(|(_, v)| v.trim().to_string()).unwrap_or_default(),
                ))
                .unwrap_or_default();

            let body = format!("{{\"authorization\":\"{sent}\"}}");
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = socket.write_all(response.as_bytes());
            let _ = socket.flush();
        }
    });

    url
}

macro_rules! two_service_app {
    ($engine_url:expr, $data_url:expr) => {
        two_service_app!($engine_url, $data_url, service::InboundSecret::new(None))
    };
    ($engine_url:expr, $data_url:expr, $inbound:expr) => {
        test::init_service(
            App::new()
                .app_data(web::Data::new(JWT_SECRET.to_string()))
                .app_data(web::Data::new(engine::Proxy::new($engine_url, None)))
                .app_data(web::Data::new(data_engine::Proxy::new($data_url, None)))
                .app_data(web::Data::new($inbound))
                .app_data(web::Data::new(tools::Registry::new(vec![])))
                .configure(api::configure),
        )
        .await
    };
}

async fn answered_by(path: &str) -> String {
    let app = two_service_app!(stub("engine"), stub("data-engine"));
    let request = test::TestRequest::get()
        .uri(path)
        .insert_header(("Authorization", session(AccessLevel::Admin)))
        .to_request();

    let body: serde_json::Value = test::call_and_read_body_json(&app, request).await;
    body["answered_by"].as_str().unwrap_or_default().to_string()
}

#[actix_web::test]
async fn a_data_route_reaches_the_data_engine() {
    assert_eq!(answered_by("/data/datasets").await, "data-engine");
}

#[actix_web::test]
async fn an_engine_route_still_reaches_the_engine() {
    assert_eq!(answered_by("/models").await, "engine");
}

#[actix_web::test]
async fn each_service_is_addressed_by_its_own_name() {
    // A copy-paste that left both crates naming "engine" would compile, route
    // correctly, and mint tokens the data engine refuses — a 401 with nothing
    // obviously wrong anywhere.
    assert_ne!(engine::AUDIENCE, data_engine::AUDIENCE);
    assert_eq!(engine::AUDIENCE, "engine");
    assert_eq!(data_engine::AUDIENCE, "data-engine");
}

#[actix_web::test]
async fn a_pull_is_refused_to_a_non_admin() {
    let app = two_service_app!(stub("engine"), stub("data-engine"));
    let request = test::TestRequest::post()
        .uri("/data/datasets/pull")
        .insert_header(("Authorization", session(AccessLevel::User)))
        .set_json(serde_json::json!({ "repo": "a/b", "name": "x" }))
        .to_request();

    let response = test::call_service(&app, request).await;

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
}

// ── the reverse hop ─────────────────────────────────────────────────────────
//
// The engine asking the backend for documents the data engine owns. Same
// app_data hazard as above, one direction further round: these routes take a
// service token where every other route takes a session, and they must reach the
// data engine while carrying the person the engine named.

const INBOUND_SECRET: &str = "two-services-inbound-secret";

fn service_call(issuer: &str, acting: &auth::Caller) -> String {
    let token = token::issue_inbound_token(issuer, acting, INBOUND_SECRET).expect("issued");
    format!("Bearer {token}")
}

fn enforcing() -> service::InboundSecret {
    service::InboundSecret::new(Some(INBOUND_SECRET.to_string()))
}

#[actix_web::test]
async fn an_internal_route_reaches_the_data_engine() {
    let app = two_service_app!(stub("engine"), stub("data-engine"), enforcing());
    let request = test::TestRequest::get()
        .uri("/internal/datasets/abc/documents?offset=0&limit=2")
        .insert_header(("Authorization", service_call("engine", &auth::Caller::System)))
        .to_request();

    let body: serde_json::Value = test::call_and_read_body_json(&app, request).await;

    assert_eq!(body["answered_by"], "data-engine");
}

#[actix_web::test]
async fn the_person_the_engine_acts_for_survives_the_hop() {
    // The whole reason the engine may ask at all: a run reads what whoever
    // queued it could have read. Losing `act` here would silently widen that to
    // whatever the data engine does with an unscoped caller.
    let owner = Uuid::new_v4();
    let app = two_service_app!(stub("engine"), echoing(), enforcing());
    let request = test::TestRequest::get()
        .uri("/internal/datasets/abc")
        .insert_header(("Authorization", service_call("engine", &auth::Caller::User(owner))))
        .to_request();

    let body: serde_json::Value = test::call_and_read_body_json(&app, request).await;
    let forwarded = body["authorization"]
        .as_str()
        .and_then(|h| h.strip_prefix("Bearer "))
        .expect("a token went out");

    assert_eq!(
        token::read_service_act(forwarded).expect("readable"),
        auth::Caller::User(owner)
    );
}

#[actix_web::test]
async fn a_browser_session_does_not_open_an_internal_route() {
    // The two schemes must not be interchangeable: a session is encrypted and
    // addressed to `web`, and nothing about it should satisfy a service guard.
    let app = two_service_app!(stub("engine"), stub("data-engine"), enforcing());
    let request = test::TestRequest::get()
        .uri("/internal/datasets/abc")
        .insert_header(("Authorization", session(AccessLevel::Admin)))
        .to_request();

    let response = test::call_service(&app, request).await;

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[actix_web::test]
async fn a_service_that_may_not_ask_is_turned_away() {
    let app = two_service_app!(stub("engine"), stub("data-engine"), enforcing());
    let request = test::TestRequest::get()
        .uri("/internal/datasets/abc")
        .insert_header((
            "Authorization",
            service_call(data_engine::AUDIENCE, &auth::Caller::System),
        ))
        .to_request();

    let response = test::call_service(&app, request).await;

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[actix_web::test]
async fn an_internal_route_needs_a_token_when_a_secret_is_set() {
    let app = two_service_app!(stub("engine"), stub("data-engine"), enforcing());
    let request = test::TestRequest::get().uri("/internal/datasets/abc").to_request();

    let response = test::call_service(&app, request).await;

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[actix_web::test]
async fn browsing_datasets_is_open_to_any_signed_in_user() {
    let app = two_service_app!(stub("engine"), stub("data-engine"));
    let request = test::TestRequest::get()
        .uri("/data/datasets")
        .insert_header(("Authorization", session(AccessLevel::User)))
        .to_request();

    let response = test::call_service(&app, request).await;

    assert_eq!(response.status(), StatusCode::OK);
}

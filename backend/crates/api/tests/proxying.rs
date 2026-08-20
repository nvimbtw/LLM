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

/// A stub that answers with something other than JSON, for the one route whose
/// response is bytes.
fn stub_bytes(
    content_type: &'static str,
    body: &'static [u8],
) -> (String, std::sync::mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        for socket in listener.incoming() {
            let Ok(mut socket) = socket else { return };
            let mut buffer = [0u8; 8192];
            let read = socket.read(&mut buffer).unwrap_or(0);
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Disposition: inline; filename*=UTF-8''shot.png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(head.as_bytes());
            let _ = socket.write_all(body);
            let _ = socket.flush();
            if tx.send(String::from_utf8_lossy(&buffer[..read]).to_string()).is_err() {
                return;
            }
        }
    });

    (url, rx)
}

/// A stub that reads the whole request before answering.
///
/// The other stubs take one 8 KiB bite and reply, which is plenty for a JSON
/// body and wrong for a forwarded file: the client is still writing when the
/// socket closes, so the hop fails as a transport error and the route answers
/// 502 for reasons that have nothing to do with what is under test.
fn stub_draining(body: &'static str) -> (String, std::sync::mpsc::Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        for socket in listener.incoming() {
            let Ok(mut socket) = socket else { return };
            let Some(head) = drain(&mut socket) else { continue };
            let response = format!(
                "HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: {}
Connection: close

{body}",
                body.len()
            );
            let _ = socket.write_all(response.as_bytes());
            let _ = socket.flush();
            if tx.send(head).is_err() {
                return;
            }
        }
    });

    (url, rx)
}

/// Read headers, then exactly as many body bytes as they promise. Returns the
/// head, which is all any assertion here looks at.
fn drain(socket: &mut std::net::TcpStream) -> Option<String> {
    let mut raw = Vec::new();
    let mut chunk = [0u8; 8192];

    // Bounded by the socket closing, and by the content length once it is known.
    loop {
        let read = socket.read(&mut chunk).ok()?;
        if read == 0 {
            return None;
        }
        raw.extend_from_slice(&chunk[..read]);

        let Some(split) = find_headers_end(&raw) else { continue };
        let head = String::from_utf8_lossy(&raw[..split]).to_string();
        let length = content_length(&head).unwrap_or(0);
        if raw.len() - split >= length {
            return Some(head);
        }
    }
}

fn find_headers_end(raw: &[u8]) -> Option<usize> {
    raw.windows(4).position(|w| w == b"\r\n\r\n").map(|at| at + 4)
}

fn content_length(head: &str) -> Option<usize> {
    head.lines()
        .find_map(|line| line.to_ascii_lowercase().strip_prefix("content-length:").map(str::to_owned))
        .and_then(|value| value.trim().parse().ok())
}

/// A multipart body with one file field, hand-built so the test asserts on the
/// wire format rather than on a helper that produces it.
fn multipart(boundary: &str, filename: &str, media_type: &str, bytes: &[u8]) -> Vec<u8> {
    let mut body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {media_type}\r\n\r\n"
    )
    .into_bytes();
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    body
}

#[actix_web::test]
async fn an_upload_reaches_the_engine_as_a_file_and_not_as_json() {
    let (url, asked) = stub_bytes("application/json", br#"{"attachment_id":"7"}"#);
    let app = service!(url);
    let (who, bearer) = session(AccessLevel::User);
    let boundary = "----test-boundary";

    let response = test::call_service(
        &app,
        test::TestRequest::post()
            .uri("/attachments")
            .insert_header(("authorization", bearer))
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            ))
            .set_payload(multipart(boundary, "shot.png", "image/png", b"pixels"))
            .to_request(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::CREATED);
    let raw = asked.recv().expect("the engine was called");
    assert_eq!(request_line(&raw), "POST /attachments HTTP/1.1");
    assert!(raw.contains("multipart/form-data"), "not forwarded as a file");
    assert!(raw.contains("shot.png"), "the filename was dropped");
    assert!(matches!(acting_for(&raw), Caller::User(id) if id == who));
}

#[actix_web::test]
async fn a_download_keeps_what_the_engine_said_the_file_is() {
    let (url, _asked) = stub_bytes("image/png", b"\x89PNG pixels");
    let app = service!(url);
    let (_, bearer) = session(AccessLevel::User);

    let response = test::call_service(
        &app,
        test::TestRequest::get()
            .uri(&format!("/attachments/{}", Uuid::new_v4()))
            .insert_header(("authorization", bearer))
            .to_request(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let headers = response.headers().clone();
    assert_eq!(headers.get("content-type").unwrap(), "image/png");
    assert!(
        headers.get("content-disposition").unwrap().to_str().unwrap().contains("shot.png"),
        "the filename the engine set was dropped"
    );
    assert_eq!(test::read_body(response).await.as_ref(), b"\x89PNG pixels");
}

#[actix_web::test]
async fn a_file_just_under_the_ceiling_still_gets_through() {
    // The companion to the test above, and the one that would have caught the
    // form limit: actix caps the whole form at 50 MiB by default, so raising
    // only the field limit to 50 MB leaves a file at the ceiling refused by a
    // limit the person was never told about. See `FORM_TOTAL_BYTES`.
    let (url, asked) = stub_draining(r#"{"attachment_id":"7"}"#);
    let app = service!(url);
    let (_, bearer) = session(AccessLevel::User);
    let boundary = "----test-boundary";
    let nearly = vec![0u8; 49 * 1024 * 1024];

    let response = test::call_service(
        &app,
        test::TestRequest::post()
            .uri("/attachments")
            .insert_header(("authorization", bearer))
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            ))
            .set_payload(multipart(boundary, "big.png", "image/png", &nearly))
            .to_request(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::CREATED);
    assert!(asked.recv().is_ok(), "it never reached the engine");
}

#[actix_web::test]
async fn an_empty_upload_never_reaches_the_engine() {
    let (url, asked) = stub("{}");
    let app = service!(url);
    let (_, bearer) = session(AccessLevel::User);
    let boundary = "----test-boundary";

    let response = test::call_service(
        &app,
        test::TestRequest::post()
            .uri("/attachments")
            .insert_header(("authorization", bearer))
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            ))
            .set_payload(multipart(boundary, "empty.txt", "text/plain", b""))
            .to_request(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(asked.try_recv().is_err(), "the engine was called anyway");
}

#[actix_web::test]
async fn an_oversized_upload_says_what_the_limit_was() {
    // Actix refuses the field while it is still reading the body, so this never
    // reaches the route — and without the error handler in `attachments.rs` it
    // comes back as a bare 400 "Payload error", which tells a person nothing
    // about why the picture they chose would not go.
    let (url, asked) = stub("{}");
    let app = service!(url);
    let (_, bearer) = session(AccessLevel::User);
    let boundary = "----test-boundary";
    let too_big = vec![0u8; 51 * 1024 * 1024];

    let response = test::call_service(
        &app,
        test::TestRequest::post()
            .uri("/attachments")
            .insert_header(("authorization", bearer))
            .insert_header((
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            ))
            .set_payload(multipart(boundary, "huge.png", "image/png", &too_big))
            .to_request(),
    )
    .await;

    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let body: Value = test::read_body_json(response).await;
    assert!(
        body["error"].as_str().unwrap_or_default().contains("50 MB"),
        "the refusal does not name the limit: {body}"
    );
    assert!(asked.try_recv().is_err(), "the engine was called anyway");
}

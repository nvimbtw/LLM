//! `/internal/gym/*` — the seam an engine-driven rollout reaches the sandbox
//! through.
//!
//! The browser's socket loop is covered by `gym_rounds.rs`; what is under test
//! here is the *other* driver's half, and specifically the one thing that
//! justifies the round trip at all: the LLM engine never sees a
//! `{stdout, stderr, exit_code}` triple, because this side renders the
//! observation with the same function chat renders one with.
//!
//! And the gate. These routes take a service token and only a service token —
//! a browser session must not reach them, or the admin-only `/gym` scope is a
//! door with a second entrance nobody checked.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc::{channel, Receiver};

use actix_web::{web, App};
use auth::token;
use auth::AccessLevel;
use serde_json::{json, Value};
use tools::Registry;
use uuid::Uuid;

const JWT_SECRET: &str = "internal-gym-test-jwt-secret";
const SERVICE_SECRET: &str = "internal-gym-test-service-secret";
const EPISODE_ID: &str = "22222222-2222-4222-8222-222222222222";
const RUN_ID: &str = "11111111-1111-4111-8111-111111111111";

/// The gyms engine, answering the two shapes this seam forwards.
fn gyms_stub() -> (String, Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("addr"));
    let (tx, rx) = channel();

    std::thread::spawn(move || {
        for socket in listener.incoming() {
            let Ok(mut socket) = socket else { return };
            let mut buffer = [0u8; 8192];
            let read = socket.read(&mut buffer).unwrap_or(0);
            let request = String::from_utf8_lossy(&buffer[..read]).to_string();

            let body = if request.contains("/step") {
                json!({ "stdout": "report.txt", "stderr": "", "exit_code": 0,
                        "timed_out": false, "step_num": 3, "reward": 0.08 })
            } else if request.contains("/finish") {
                json!({ "episode_id": EPISODE_ID, "score": 1.0, "correct": true,
                        "answer": "report.txt", "ground_truth": "report.txt" })
            } else if request.contains("/gym/runs") {
                json!({ "id": RUN_ID, "kind": "rl", "episodes": [] })
            } else {
                json!({ "ok": true })
            }
            .to_string();

            let _ = socket.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            );
            let _ = socket.flush();
            if tx.send(request).is_err() {
                return;
            }
        }
    });

    (url, rx)
}

fn server(gyms_url: String) -> actix_test::TestServer {
    actix_test::start(move || {
        App::new()
            .app_data(web::Data::new(JWT_SECRET.to_string()))
            .app_data(web::Data::new(service::InboundSecret::new(Some(
                SERVICE_SECRET.to_string(),
            ))))
            .app_data(web::Data::new(gyms_engine::Proxy::new(
                gyms_url.clone(),
                Some(SERVICE_SECRET.to_string()),
            )))
            .app_data(web::Data::new(Registry::new(vec![])))
            .configure(api::configure)
    })
}

/// What the LLM engine sends: `aud: backend`, signed with the shared secret. The
/// mirror of the token this server mints going the other way.
fn service_token() -> String {
    token::issue_inbound_token("llm-engine", &auth::Caller::System, SERVICE_SECRET).expect("minted")
}

fn admin_token() -> String {
    let jwt =
        token::issue_user_token(Uuid::new_v4(), AccessLevel::Admin, JWT_SECRET).expect("issued");
    auth::token_cipher::encrypt(&jwt, JWT_SECRET).expect("sealed")
}

#[actix_web::test]
async fn the_engine_can_read_the_catalogue_but_a_browser_token_cannot() {
    let (gyms_url, served) = gyms_stub();
    let server = server(gyms_url);
    let response = server
        .get("/internal/gyms")
        .insert_header(("Authorization", format!("Bearer {}", service_token())))
        .send()
        .await
        .expect("sent");
    assert_eq!(response.status(), 200);
    assert!(served.recv().expect("forwarded").contains("GET /gyms "));
    let refused = server
        .get("/internal/gyms")
        .insert_header(("Authorization", format!("Bearer {}", admin_token())))
        .send()
        .await
        .expect("sent");
    assert_eq!(refused.status(), 401);
    assert!(served.try_recv().is_err());
}

#[actix_web::test]
async fn a_step_comes_back_rendered_so_the_engine_never_lays_one_out() {
    let (gyms_url, _served) = gyms_stub();
    let server = server(gyms_url);

    let mut response = server
        .post("/internal/gym/episodes/{id}/step".replace("{id}", EPISODE_ID))
        .insert_header(("Authorization", format!("Bearer {}", service_token())))
        .send_json(&json!({ "tool": "bash", "argument": "ls -la", "reply": "let me look" }))
        .await
        .expect("sent");

    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.expect("json");

    // The rendered block, not the triple. A `<|codeblock|>` is what
    // `tools::observation` produces and what chat has always shown the model.
    let observation = body["observation"].as_str().expect("an observation");
    assert!(observation.contains("report.txt"), "{observation}");
    assert!(observation.contains("<|codeblock|>"), "{observation}");

    // The shaping reward and the step number ride along: they are what the
    // rollout's return is built from, and a second request for them would be a
    // second round trip per command.
    assert_eq!(body["reward"], json!(0.08));
    assert_eq!(body["step_num"], json!(3));
}

#[actix_web::test]
async fn a_tool_this_gym_does_not_have_is_a_caveat_rather_than_a_step() {
    let (gyms_url, served) = gyms_stub();
    let server = server(gyms_url);

    let mut response = server
        .post(format!("/internal/gym/episodes/{EPISODE_ID}/step"))
        .insert_header(("Authorization", format!("Bearer {}", service_token())))
        .send_json(&json!({ "tool": "web-search", "argument": "tallest mountain" }))
        .await
        .expect("sent");

    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.expect("json");
    assert!(
        body["observation"]
            .as_str()
            .unwrap_or_default()
            .contains("no web-search tool"),
        "{body}"
    );
    // Nothing ran, so nothing was recorded and no reward was earned. A gym that
    // charged a step for a tool it does not have would be paying for the model's
    // mistake out of the episode's round budget.
    assert!(body["reward"].is_null(), "{body}");
    assert!(
        served.try_recv().is_err(),
        "the gyms engine was asked to run it"
    );
}

#[actix_web::test]
async fn a_browser_session_cannot_reach_the_rollout_seam() {
    let (gyms_url, _served) = gyms_stub();
    let server = server(gyms_url);

    let response = server
        .post(format!("/internal/gym/episodes/{EPISODE_ID}/step"))
        .insert_header(("Authorization", format!("Bearer {}", admin_token())))
        .send_json(&json!({ "tool": "bash", "argument": "ls" }))
        .await
        .expect("sent");

    // An admin's own token, refused. `ServiceCaller` accepts one issuer and the
    // browser is not it — being an admin is what opens `/gym`, not this.
    assert_eq!(response.status(), 401);
}

#[actix_web::test]
async fn grading_and_creating_pass_through_untouched() {
    let (gyms_url, served) = gyms_stub();
    let server = server(gyms_url);

    let mut graded = server
        .post(format!("/internal/gym/episodes/{EPISODE_ID}/finish"))
        .insert_header(("Authorization", format!("Bearer {}", service_token())))
        .send_json(&json!({ "reply": "report.txt" }))
        .await
        .expect("sent");
    assert_eq!(graded.status(), 200);
    assert_eq!(
        graded.json::<Value>().await.expect("json")["score"],
        json!(1.0)
    );

    let created = server
        .post("/internal/gym/runs")
        .insert_header(("Authorization", format!("Bearer {}", service_token())))
        .send_json(&json!({ "gym_id": "bash", "count": 4, "kind": "rl" }))
        .await
        .expect("sent");
    assert_eq!(created.status(), 200);

    // `kind` is forwarded rather than set here: a batch that wants to stay off
    // the evaluation board says so, and a proxy rewriting the body it forwards
    // is worse than one that does not.
    let forwarded: Vec<String> = (0..2).map(|_| served.recv().expect("a request")).collect();
    assert!(forwarded[1].contains("\"kind\":\"rl\""), "{}", forwarded[1]);
}

//! Verify the Rust authorization and opaque transport seam independently of Python execution.

use actix_web::{
    http::{Method, StatusCode},
    test, web, App, HttpRequest, HttpResponse,
};
use auth::{token, AccessLevel};
use serde_json::{json, Value};
use uuid::Uuid;

const SECRET: &str = "architecture-proxy-browser-secret";
const SERVICE_SECRET: &str = "architecture-proxy-service-secret";

/// Issue an encrypted browser credential for a known actor and access level.
fn session(actor: Uuid, level: AccessLevel) -> String {
    let jwt = token::issue_user_token(actor, level, SECRET).unwrap();
    format!(
        "Bearer {}",
        auth::token_cipher::encrypt(&jwt, SECRET).unwrap()
    )
}

/// Verify the forwarded token and echo its trusted identity, scope, request path and body.
async fn echo(req: HttpRequest, body: web::Bytes) -> HttpResponse {
    let token = req
        .headers()
        .get("Authorization")
        .unwrap()
        .to_str()
        .unwrap()
        .strip_prefix("Bearer ")
        .unwrap();
    let actor = token::verify_service_token(token, "llm-engine", SERVICE_SECRET).unwrap();
    let mut validation = jsonwebtoken::Validation::default();
    validation.set_audience(&["llm-engine"]);
    let claims = jsonwebtoken::decode::<token::ServiceClaims>(
        token,
        &jsonwebtoken::DecodingKey::from_secret(SERVICE_SECRET.as_bytes()),
        &validation,
    )
    .unwrap()
    .claims;
    let status = match req
        .query_string()
        .split('&')
        .find(|value| matches!(*value, "created" | "queued" | "conflict"))
    {
        Some("created") => StatusCode::CREATED,
        Some("queued") => StatusCode::ACCEPTED,
        Some("conflict") => StatusCode::CONFLICT,
        _ => StatusCode::OK,
    };
    HttpResponse::build(status).json(json!({
        "actor": match actor { auth::Caller::User(id) => id.to_string(), _ => "system".into() },
        "scope": claims.scope,
        "path": req.uri().to_string(),
        "body": serde_json::from_slice::<Value>(&body).unwrap_or(Value::Null),
        "error": {"phase":"capture", "path":"blocks.core", "message":"structured diagnostic"},
    }))
}

/// Every source, draft, graph, report and catalog action rejects nonadministrators before upstream access.
#[actix_web::test]
async fn private_routes_require_administration() {
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(SECRET.to_string()))
            .app_data(web::Data::new(llm_engine::Proxy::new(
                "http://127.0.0.1:1",
                None,
            )))
            .configure(api::configure),
    )
    .await;
    let id = Uuid::new_v4();
    let mut routes = vec![
        (Method::POST, "/runs/train".into()),
        (Method::POST, "/runs/sft".into()),
        (Method::POST, "/runs/rl".into()),
        (Method::POST, "/runs/chain".into()),
        (Method::POST, format!("/runs/{id}/retry")),
        (Method::POST, format!("/runs/{id}/recover")),
        (Method::POST, format!("/runs/{id}/cancel")),
        (Method::PATCH, format!("/runs/{id}")),
        (Method::DELETE, format!("/runs/{id}")),
        (Method::GET, "/architectures/schema".into()),
        (Method::POST, format!("/architectures/{id}/declaration")),
        (Method::GET, format!("/architectures/revisions/{id}")),
        (
            Method::POST,
            format!("/architectures/revisions/{id}/training-preview"),
        ),
        (Method::GET, format!("/validations/{id}")),
        (Method::GET, format!("/validations/{id}/graph")),
        (Method::GET, format!("/models/{id}/definition")),
        (Method::GET, format!("/models/{id}/graph")),
        (Method::GET, format!("/models/{id}/sources/{id}")),
        (Method::GET, format!("/validations/{id}/sources/{id}")),
        (
            Method::GET,
            format!("/tokenizers/{id}/architecture-binding"),
        ),
        (
            Method::GET,
            format!("/blocks/{id}/revisions/{id}/copy-source"),
        ),
    ];
    for catalog in ["blocks", "architectures"] {
        for (method, suffix) in [
            (Method::GET, "".to_string()),
            (Method::POST, "".to_string()),
            (Method::GET, format!("/{id}")),
            (Method::PATCH, format!("/{id}")),
            (Method::GET, format!("/{id}/draft")),
            (Method::PUT, format!("/{id}/draft")),
            (Method::GET, format!("/{id}/revisions")),
            (Method::POST, format!("/{id}/revisions")),
            (Method::GET, format!("/{id}/revisions/{id}")),
            (Method::POST, format!("/{id}/validations")),
        ] {
            routes.push((method, format!("/{catalog}{suffix}")));
        }
    }
    for (method, path) in routes {
        for authorized in [false, true] {
            let mut req = test::TestRequest::default()
                .method(method.clone())
                .uri(&path)
                .set_json(json!({}));
            if authorized {
                req = req.insert_header(("Authorization", session(id, AccessLevel::User)));
            }
            let response = test::call_service(&app, req.to_request()).await;
            assert_eq!(
                response.status(),
                if authorized {
                    StatusCode::FORBIDDEN
                } else {
                    StatusCode::UNAUTHORIZED
                },
                "{method} {path}"
            );
        }
    }
}

/// Catalog requests retain actor, audience, body, query, structured errors and success statuses.
#[actix_web::test]
async fn opaque_forwarding_preserves_status_and_identity() {
    let server = actix_test::start(|| App::new().default_service(web::to(echo)));
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(SECRET.to_string()))
            .app_data(web::Data::new(llm_engine::Proxy::new(
                server.url("").trim_end_matches('/'),
                Some(SERVICE_SECRET.into()),
            )))
            .configure(api::configure),
    )
    .await;
    let actor = Uuid::new_v4();
    for (query, status) in [("created", 201), ("queued", 202), ("conflict", 409)] {
        let req = test::TestRequest::get()
            .uri(&format!("/architectures/schema?{query}"))
            .insert_header(("Authorization", session(actor, AccessLevel::Admin)))
            .to_request();
        let response = test::call_service(&app, req).await;
        assert_eq!(response.status().as_u16(), status);
        let body: Value = test::read_body_json(response).await;
        assert_eq!(body["actor"], actor.to_string());
        assert_eq!(body["scope"], json!(["administration"]));
        assert_eq!(body["error"]["phase"], "capture");
    }
    let fixtures: Value = serde_json::from_str(include_str!(
        "../../../../contracts/modular-architecture/v1/api/cases.json"
    ))
    .unwrap();
    let payload = fixtures["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "BlockDraftSave-positive")
        .unwrap()["value"]
        .clone();
    let path = format!("/blocks/{actor}/draft");
    let req = test::TestRequest::put()
        .uri(&path)
        .set_json(&payload)
        .insert_header(("Authorization", session(actor, AccessLevel::Admin)))
        .to_request();
    let body: Value = test::call_and_read_body_json(&app, req).await;
    assert_eq!(body["path"], path);
    assert_eq!(body["body"], payload);
}

/// Crafted browser filters cannot elevate the trusted role used for pagination or metadata reads.
#[actix_web::test]
async fn visibility_is_signed_from_verified_claims() {
    let server = actix_test::start(|| App::new().default_service(web::to(echo)));
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(SECRET.to_string()))
            .app_data(web::Data::new(llm_engine::Proxy::new(
                server.url("").trim_end_matches('/'),
                Some(SERVICE_SECRET.into()),
            )))
            .configure(api::configure),
    )
    .await;
    let actor = Uuid::new_v4();
    for path in [
        "/runs?visibility=administration&scope=administration&limit=2&offset=4",
        "/models?scope=administration",
    ] {
        let req = test::TestRequest::get()
            .uri(path)
            .insert_header(("Authorization", session(actor, AccessLevel::User)))
            .to_request();
        let body: Value = test::call_and_read_body_json(&app, req).await;
        assert_eq!(body["scope"], json!(["public"]));
        assert_eq!(body["actor"], actor.to_string());
        assert_eq!(body["path"], path);
    }
}

/// Forward recipe submission/retry identity without losing actor, administrative scope or upstream status.
#[actix_web::test]
async fn training_writes_preserve_queries_actor_scope_and_status() {
    let server = actix_test::start(|| App::new().default_service(web::to(echo)));
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(SECRET.to_string()))
            .app_data(web::Data::new(llm_engine::Proxy::new(
                server.url("").trim_end_matches('/'),
                Some(SERVICE_SECRET.into()),
            )))
            .configure(api::configure),
    )
    .await;
    let actor = Uuid::new_v4();
    let operation = Uuid::new_v4();
    let paths = [
        ("/runs/train".to_string(), "/train".to_string()),
        ("/runs/sft".to_string(), "/sft".to_string()),
        ("/runs/rl".to_string(), "/rl".to_string()),
        ("/runs/chain".to_string(), "/runs/chain".to_string()),
        (
            format!("/runs/{actor}/cancel"),
            format!("/runs/{actor}/cancel"),
        ),
        (
            format!("/runs/{actor}/retry"),
            format!("/runs/{actor}/retry"),
        ),
        (
            format!("/runs/{actor}/recover"),
            format!("/runs/{actor}/recover"),
        ),
    ];
    let payload = json!({"architecture":{"revision_id":actor,"parameters":{"repeats":3}}});
    for (browser, upstream) in paths {
        let query = format!("queued&request_id={operation}");
        let req = test::TestRequest::post()
            .uri(&format!("{browser}?{query}"))
            .insert_header(("Authorization", session(actor, AccessLevel::Admin)))
            .set_json(&payload)
            .to_request();
        let response = test::call_service(&app, req).await;
        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let body: Value = test::read_body_json(response).await;
        assert_eq!(body["actor"], actor.to_string());
        assert_eq!(body["scope"], json!(["administration"]));
        assert_eq!(body["path"], format!("{upstream}?{query}"));
        if matches!(
            browser.as_str(),
            "/runs/train" | "/runs/sft" | "/runs/rl" | "/runs/chain"
        ) {
            assert_eq!(body["body"], payload);
        } else {
            assert_eq!(body["body"], Value::Null);
        }
    }
}

/// Request admission caps encoded JSON even when the administrator submits a huge source field.
#[actix_web::test]
async fn oversized_draft_never_reaches_the_service() {
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(SECRET.to_string()))
            .app_data(web::Data::new(llm_engine::Proxy::new(
                "http://127.0.0.1:1",
                None,
            )))
            .configure(api::configure),
    )
    .await;
    let req = test::TestRequest::post()
        .uri("/blocks")
        .insert_header(("Authorization", session(Uuid::new_v4(), AccessLevel::Admin)))
        .set_json(json!({"source":"x".repeat(1024*1024)}))
        .to_request();
    assert_eq!(
        test::call_service(&app, req).await.status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
}

/// Admit administration event streams and refuse public observation before emitting private data.
async fn events(req: HttpRequest) -> HttpResponse {
    let token = req
        .headers()
        .get("Authorization")
        .unwrap()
        .to_str()
        .unwrap()
        .strip_prefix("Bearer ")
        .unwrap();
    token::verify_service_token(token, "llm-engine", SERVICE_SECRET).unwrap();
    let mut validation = jsonwebtoken::Validation::default();
    validation.set_audience(&["llm-engine"]);
    let claims = jsonwebtoken::decode::<token::ServiceClaims>(
        token,
        &jsonwebtoken::DecodingKey::from_secret(SERVICE_SECRET.as_bytes()),
        &validation,
    )
    .unwrap()
    .claims;
    if claims.scope != ["administration"] {
        return HttpResponse::Forbidden().json(json!({"detail":"administration required"}));
    }
    HttpResponse::Ok()
        .content_type("text/event-stream")
        .body("event: validation\ndata: {\"source\":\"private block\"}\n\n")
}

/// A public socket cannot receive private validation events, while an administrator can observe them.
#[actix_web::test]
async fn job_sockets_propagate_visibility_before_event_replay() {
    use futures_util::{SinkExt, StreamExt};
    let upstream = actix_test::start(|| App::new().default_service(web::to(events)));
    let url = upstream.url("").trim_end_matches('/').to_string();
    let backend = actix_test::start(move || {
        App::new()
            .app_data(web::Data::new(SECRET.to_string()))
            .app_data(web::Data::new(llm_engine::Proxy::new(
                &url,
                Some(SERVICE_SECRET.into()),
            )))
            .configure(api::configure)
    });
    for (level, private) in [(AccessLevel::User, false), (AccessLevel::Admin, true)] {
        let (_, mut socket) = awc::Client::new()
            .ws(backend.url(&format!("/ws/runs/{}", Uuid::new_v4())))
            .connect()
            .await
            .unwrap();
        let bearer = session(Uuid::new_v4(), level);
        socket
            .send(awc::ws::Message::Text(
                json!({"token":bearer.strip_prefix("Bearer ").unwrap()})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        let ready = socket.next().await.unwrap().unwrap();
        assert!(matches!(ready, awc::ws::Frame::Text(_)));
        let frame = socket.next().await.unwrap().unwrap();
        let awc::ws::Frame::Text(text) = frame else {
            panic!("expected event/error frame")
        };
        let text = String::from_utf8(text.to_vec()).unwrap();
        assert_eq!(text.contains("private block"), private);
        if !private {
            assert!(text.contains("error"));
        }
        socket.close().await.unwrap();
    }
}

/// Bounded JSON rejects over-budget or undecodable responses without echoing sensitive raw bytes.
#[actix_web::test]
async fn response_limits_and_decode_errors_hide_raw_upstream_data() {
    let server = actix_test::start(|| {
        App::new().default_service(web::to(|| async {
            HttpResponse::Ok().body("private source: invalid JSON and deliberately over budget")
        }))
    });
    let proxy = llm_engine::Proxy::new(
        server.url("").trim_end_matches('/'),
        Some(SERVICE_SECRET.into()),
    );
    for limit in [8, 1024] {
        let error = proxy
            .scoped_json(
                &auth::Caller::User(Uuid::new_v4()),
                "administration",
                llm_engine::Method::GET,
                "/blocks",
                None,
                limit,
            )
            .await
            .unwrap_err();
        assert!(!error.to_string().contains("private source"));
        assert!(matches!(error, service::Error::Decode(_)));
    }
}

/// Own the disposable Python service, storage and its complete termination on test exit.
struct LiveEngine {
    child: std::process::Child,
    root: std::path::PathBuf,
    manifest: Value,
}

impl LiveEngine {
    /// Start the real engine routers with isolated fixtures and no production application lifespan.
    fn start() -> Self {
        use std::io::BufRead;
        let repository = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        let root = std::env::temp_dir().join(format!("slate-rust-visibility-{}", Uuid::new_v4()));
        std::fs::create_dir(&root).unwrap();
        let python = std::env::var_os("SLATE_ENGINE_PYTHON")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| repository.join("engine/.venv/bin/python"));
        let child = std::process::Command::new(python)
            .arg(repository.join("backend/crates/api/tests/fixtures/visibility_engine.py"))
            .arg(&root)
            .arg(repository.join("llm-engine"))
            .env("LLM_ENGINE_SERVICE_SECRET", SERVICE_SECRET)
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut engine = Self {
            child,
            root,
            manifest: Value::Null,
        };
        let mut line = String::new();
        std::io::BufReader::new(engine.child.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        engine.manifest = serde_json::from_str(&line).expect("Python fixture startup manifest");
        engine
    }
}

impl Drop for LiveEngine {
    /// Stop/reap the Python process before removing only this test's owned scratch directory.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// Real Rust credentials reach Python filters before paging, metadata serialization and socket replay.
#[actix_web::test]
#[ignore = "requires installed engine Python dependencies; run explicitly for the live Rust/Python seam"]
async fn live_python_visibility_http_and_websocket_contract() {
    use futures_util::{SinkExt, StreamExt};
    let engine = LiveEngine::start();
    let url = engine.manifest["url"].as_str().unwrap().to_string();
    let upstream_url = url.clone();
    let bench = actix_test::start(|| {
        App::new().default_service(web::to(|| async {
            HttpResponse::Ok().json(json!({"benchmarks": []}))
        }))
    });
    let bench_url = bench.url("").trim_end_matches('/').to_string();
    let backend = actix_test::start(move || {
        App::new()
            .app_data(web::Data::new(SECRET.to_string()))
            .app_data(web::Data::new(llm_engine::Proxy::new(
                &upstream_url,
                Some(SERVICE_SECRET.into()),
            )))
            .app_data(web::Data::new(bench_engine::Proxy::new(&bench_url, None)))
            .configure(api::configure)
    });
    let actor = Uuid::new_v4();
    let public = session(actor, AccessLevel::User);
    let admin = session(actor, AccessLevel::Admin);
    let client = awc::Client::default();
    for _ in 0..100 {
        if client.get(format!("{url}/runs")).send().await.is_ok() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    for (bearer, total, active) in [(&public, 3, 0), (&admin, 7, 1)] {
        let mut response = client
            .get(backend.url("/runs?limit=2&scope=administration&visibility=administration"))
            .insert_header(("Authorization", bearer.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["history_total"], total);
        assert_eq!(body["active"].as_array().unwrap().len(), active);
        assert_eq!(body["history"].as_array().unwrap().len(), 2);
        assert_eq!(
            body.to_string().contains("PRIVATE_SOURCE_GRAPH_MARKER"),
            total == 7
        );
    }
    let mut second = client
        .get(backend.url("/runs?limit=2&offset=2"))
        .insert_header(("Authorization", public.as_str()))
        .send()
        .await
        .unwrap();
    let body: Value = second.json().await.unwrap();
    assert_eq!(body["history"].as_array().unwrap().len(), 1);
    let validation = engine.manifest["validation_0"].as_str().unwrap();
    for (bearer, status) in [(&public, StatusCode::FORBIDDEN), (&admin, StatusCode::OK)] {
        let mut response = client
            .get(backend.url(&format!("/runs/{validation}")))
            .insert_header(("Authorization", bearer.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), status);
        let body = response.body().await.unwrap();
        assert_eq!(
            String::from_utf8_lossy(&body).contains("PRIVATE_SOURCE_GRAPH_MARKER"),
            status == StatusCode::OK
        );
    }
    let model = engine.manifest["model"].as_str().unwrap();
    for path in [
        "/models".to_string(),
        format!("/models/{model}"),
        format!("/models/{model}/checkpoints"),
    ] {
        let mut response = client
            .get(backend.url(&path))
            .insert_header(("Authorization", public.as_str()))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(!String::from_utf8_lossy(&response.body().await.unwrap())
            .contains("PRIVATE_SOURCE_GRAPH_MARKER"));
    }
    let proxy = llm_engine::Proxy::new(&url, Some(SERVICE_SECRET.into()));
    let refusal = proxy
        .scoped_events(
            &auth::Caller::User(actor),
            "public",
            &format!("/runs/{validation}/events"),
        )
        .await;
    assert!(matches!(
        refusal,
        Err(service::Error::Status { code: 403, .. })
    ));
    let public_run = engine.manifest["public_0"].as_str().unwrap();
    let mut frames = Box::pin(
        proxy
            .scoped_events(
                &auth::Caller::User(actor),
                "public",
                &format!("/runs/{public_run}/events"),
            )
            .await
            .unwrap(),
    );
    let mut observed = Vec::new();
    while let Some(frame) = frames.next().await {
        observed.push(frame.unwrap().data);
    }
    let serialized = serde_json::to_string(&observed).unwrap();
    assert!(!serialized.contains("PRIVATE_SOURCE_GRAPH_MARKER"));
    assert!(serialized.contains("1.25"));
    for (bearer, private) in [(&public, false), (&admin, true)] {
        let (_, mut socket) = client
            .ws(backend.url(&format!("/ws/runs/{validation}")))
            .connect()
            .await
            .unwrap();
        socket
            .send(awc::ws::Message::Text(
                json!({"token":bearer.strip_prefix("Bearer ").unwrap()})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        let _ = socket.next().await.unwrap().unwrap();
        let awc::ws::Frame::Text(frame) = socket.next().await.unwrap().unwrap() else {
            panic!("event frame")
        };
        let text = String::from_utf8_lossy(&frame);
        assert_eq!(text.contains("PRIVATE_SOURCE_GRAPH_MARKER"), private);
        if !private {
            assert!(text.contains("error"));
        }
        socket.close().await.unwrap();
    }
}

/// Attention discovery preserves ordinary-user actor/scope and exact service failures without exposing authoring routes.
#[actix_web::test]
async fn attention_discovery_is_a_public_authenticated_read() {
    let server = actix_test::start(|| App::new().default_service(web::to(echo)));
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(SECRET.to_string()))
            .app_data(web::Data::new(llm_engine::Proxy::new(
                server.url("").trim_end_matches('/'),
                Some(SERVICE_SECRET.into()),
            )))
            .configure(api::configure),
    )
    .await;
    let actor = Uuid::new_v4();
    let model = Uuid::new_v4();
    let path = format!("/models/{model}/attention-points");
    let response = test::call_service(&app, test::TestRequest::get().uri(&path).to_request()).await;
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    for (level, scope) in [
        (AccessLevel::User, "public"),
        (AccessLevel::Admin, "administration"),
    ] {
        let response = test::call_service(
            &app,
            test::TestRequest::get()
                .uri(&path)
                .insert_header(("Authorization", session(actor, level)))
                .to_request(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value = test::read_body_json(response).await;
        assert_eq!(body["actor"], actor.to_string());
        assert_eq!(body["scope"], json!([scope]));
        assert_eq!(body["path"], path);
    }
}

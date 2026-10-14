//! Private project HTTP boundaries and backend-owned memory context across inference transports.

use actix_web::{http::StatusCode, test, web, App, HttpResponse};
use auth::{token, token_cipher, AccessLevel};
use futures_util::{SinkExt, StreamExt};
use llm_engine::Proxy;
use serde_json::{json, Value};
use sqlx::PgPool;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

const SECRET: &str = "projects-api-fixture";
type Requests = Arc<Mutex<Vec<(String, Value)>>>;

/// Seal a test user's JWT, optionally marking it admin to exercise owner isolation despite elevated access.
fn bearer(owner: Uuid, admin: bool) -> String {
    let access = if admin {
        AccessLevel::Admin
    } else {
        AccessLevel::User
    };
    let token = token::issue_user_token(owner, access, SECRET).unwrap();
    format!("Bearer {}", token_cipher::encrypt(&token, SECRET).unwrap())
}

/// Insert a fixture user with a distinct username and return its durable identity.
async fn owner(pool: &PgPool) -> Uuid {
    db::user::create(pool, &Uuid::new_v4().to_string(), "hash", None)
        .await
        .unwrap()
        .id_user
}

/// Build an independent stored fixture message with the supplied role and content.
fn message(role: &str, content: &str) -> db::conversation::NewMessage {
    db::conversation::NewMessage {
        role: role.into(),
        content: content.into(),
        tokens: 4,
        model_id: None,
        model_name: None,
        tool: None,
        argument: None,
        request_id: Uuid::new_v4(),
    }
}

/// Record an engine request under its protocol operation, retaining the parsed body for assertions.
fn record(log: &Requests, operation: &str, body: &Value) {
    log.lock().unwrap().push((operation.into(), body.clone()));
}

/// Canonically prepare the fixture's user/tool input and optional steering while recording the transport payload.
async fn prepare(body: web::Json<Value>, log: web::Data<Requests>) -> HttpResponse {
    record(&log, "prepare", &body);
    if body["model_name"] == "incompatible" {
        return HttpResponse::BadRequest()
            .json(json!({"detail":"memory tokens cannot be adopted"}));
    }
    let mut turns = if let Some(text) = body["text"].as_str() {
        vec![json!({"role":"user","text":text,"tokens":4})]
    } else {
        vec![json!({"role":"tool","text":"canonical tool result","tokens":4})]
    };
    if let Some(text) = body["interject"].as_str() {
        turns.push(json!({"role":"user","text":text,"tokens":4}));
    }
    HttpResponse::Ok().json(json!({"turns":turns,"model_name":"tiny","model_id":null}))
}

/// Record a stateless HTTP generation request and return a complete persistable assistant answer.
async fn generate(body: web::Json<Value>, log: web::Data<Requests>) -> HttpResponse {
    record(&log, "generate", &body);
    HttpResponse::Ok().json(json!({"text":"answer","generated_tokens":4}))
}

/// Record context-floor inputs and keep all fixture history in the engine's response.
async fn floor(body: web::Json<Value>, log: web::Data<Requests>) -> HttpResponse {
    record(&log, "floor", &body);
    HttpResponse::Ok().json(json!({"start":0}))
}

/// Record a streamed generation payload and emit one final answer without any tool continuation.
async fn stream(body: web::Json<Value>, log: web::Data<Requests>) -> HttpResponse {
    record(&log, "stream", &body);
    HttpResponse::Ok()
        .content_type("text/event-stream")
        .body("event: done\ndata: {\"text\":\"answer\",\"generated_tokens\":4}\n\n")
}

/// Run an isolated fixture engine that records real HTTP calls from the backend proxy.
fn engine(log: Requests) -> actix_test::TestServer {
    actix_test::start(move || {
        App::new()
            .app_data(web::Data::new(log.clone()))
            .route("/chat/prepare", web::post().to(prepare))
            .route("/chat", web::post().to(generate))
            .route("/chat/context-floor", web::post().to(floor))
            .route("/chat/stream", web::post().to(stream))
    })
}

/// Exercise CRUD, private admin isolation, atomic assignment, pagination and deletion without any engine.
#[sqlx::test(migrations = "../db/migrations")]
async fn projects_are_private_and_delete_preserves_history(pool: PgPool) {
    let alice = owner(&pool).await;
    let bob = owner(&pool).await;
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(pool.clone()))
            .app_data(web::Data::new(SECRET.to_owned()))
            .configure(api::configure),
    )
    .await;
    let request = test::TestRequest::post()
        .uri("/projects")
        .insert_header(("Authorization", bearer(alice, false)))
        .set_json(json!({"name":"Orbit","description":"battery plans","notes":"use metric units"}))
        .to_request();
    let response = test::call_service(&app, request).await;
    assert_eq!(response.status(), StatusCode::CREATED);
    let project: Value = test::read_body_json(response).await;
    let id = project["project_id"].as_str().unwrap();
    let path = format!("/projects/{id}");
    let request = test::TestRequest::post()
        .uri("/chats")
        .insert_header(("Authorization", bearer(alice, false)))
        .set_json(json!({"title":"linked","project_id":id}))
        .to_request();
    let chat: Value = test::call_and_read_body_json(&app, request).await;
    assert_eq!(chat["project_id"], id);
    let chat_id = Uuid::parse_str(chat["chat_id"].as_str().unwrap()).unwrap();
    let stored =
        db::conversation::append(&pool, alice, chat_id, &message("user", "orbit input"), &[])
            .await
            .unwrap();
    for request in [
        test::TestRequest::get().uri(&path),
        test::TestRequest::patch()
            .uri(&path)
            .set_json(json!({"name":"stolen"})),
        test::TestRequest::delete().uri(&path),
        test::TestRequest::post()
            .uri("/chats")
            .set_json(json!({"project_id":id})),
        test::TestRequest::put()
            .uri(&format!("/chats/{chat_id}/project"))
            .set_json(json!({"project_id":id})),
    ] {
        let request = request
            .insert_header(("Authorization", bearer(bob, true)))
            .to_request();
        assert_eq!(
            test::call_service(&app, request).await.status(),
            StatusCode::NOT_FOUND
        );
    }
    let request = test::TestRequest::get()
        .uri("/projects")
        .insert_header(("Authorization", bearer(bob, true)))
        .to_request();
    let listing: Value = test::call_and_read_body_json(&app, request).await;
    assert_eq!(listing["projects"], json!([]));
    let request = test::TestRequest::get()
        .uri(&format!("{path}?limit=1&offset=0"))
        .insert_header(("Authorization", bearer(alice, false)))
        .to_request();
    let detail: Value = test::call_and_read_body_json(&app, request).await;
    assert_eq!(detail["threads_total"], 1);
    assert_eq!(detail["thread_count"], detail["threads_total"]);
    assert_eq!(detail["threads_limit"], 1);
    assert_eq!(detail["threads"][0]["chat_id"], chat["chat_id"]);
    let request = test::TestRequest::get()
        .uri(&format!("{path}?limit=1&offset=1"))
        .insert_header(("Authorization", bearer(alice, false)))
        .to_request();
    let beyond: Value = test::call_and_read_body_json(&app, request).await;
    assert_eq!(beyond["thread_count"], 1);
    assert_eq!(beyond["threads_total"], 1);
    assert_eq!(beyond["threads_offset"], 1);
    assert_eq!(beyond["threads"], json!([]));
    let request = test::TestRequest::patch()
        .uri(&path)
        .insert_header(("Authorization", bearer(alice, false)))
        .set_json(json!({"notes":"updated instructions"}))
        .to_request();
    let updated: Value = test::call_and_read_body_json(&app, request).await;
    assert_eq!(updated["name"], "Orbit");
    assert_eq!(updated["notes"], "updated instructions");
    let request = test::TestRequest::put()
        .uri(&format!("/chats/{chat_id}/project"))
        .insert_header(("Authorization", bearer(alice, false)))
        .set_json(json!({"project_id":null}))
        .to_request();
    let detached: Value = test::call_and_read_body_json(&app, request).await;
    assert!(detached["project_id"].is_null());
    let request = test::TestRequest::put()
        .uri(&format!("/chats/{chat_id}/project"))
        .insert_header(("Authorization", bearer(alice, false)))
        .set_json(json!({"project_id":id}))
        .to_request();
    assert_eq!(
        test::call_service(&app, request).await.status(),
        StatusCode::OK
    );
    let request = test::TestRequest::delete()
        .uri(&path)
        .insert_header(("Authorization", bearer(alice, false)))
        .to_request();
    assert_eq!(
        test::call_service(&app, request).await.status(),
        StatusCode::NO_CONTENT
    );
    let request = test::TestRequest::get()
        .uri(&format!("/chats/{chat_id}"))
        .insert_header(("Authorization", bearer(alice, false)))
        .to_request();
    let history: Value = test::call_and_read_body_json(&app, request).await;
    assert!(history["project_id"].is_null());
    assert_eq!(history["messages"][0]["message_id"], json!(stored.id));
}

/// Validate Unicode field budgets, bounded pagination and explicit typed assignment at HTTP boundaries.
#[sqlx::test(migrations = "../db/migrations")]
async fn project_boundaries_reject_invalid_and_excessive_values(pool: PgPool) {
    let alice = owner(&pool).await;
    let chat = db::conversation::create(&pool, alice, "standalone")
        .await
        .unwrap();
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(pool))
            .app_data(web::Data::new(SECRET.to_owned()))
            .configure(api::configure),
    )
    .await;
    for fields in [
        json!({}),
        json!({"name":" \n"}),
        json!({"name":7}),
        json!({"name":"x".repeat(121)}),
        json!({"name":"ok","description":"x".repeat(2001)}),
        json!({"name":"ok","notes":"x".repeat(8001)}),
        json!({"name":"ok","owner_id":Uuid::new_v4()}),
    ] {
        let request = test::TestRequest::post()
            .uri("/projects")
            .insert_header(("Authorization", bearer(alice, false)))
            .set_json(fields)
            .to_request();
        assert_eq!(
            test::call_service(&app, request).await.status(),
            StatusCode::BAD_REQUEST
        );
    }
    let request = test::TestRequest::post().uri("/projects")
        .insert_header(("Authorization", bearer(alice, false)))
        .set_json(json!({"name":"🦀".repeat(120),"description":"x".repeat(2000),"notes":"x".repeat(8000)})).to_request();
    assert_eq!(
        test::call_service(&app, request).await.status(),
        StatusCode::CREATED
    );
    for query in [
        "limit=0",
        "limit=201",
        "offset=-1",
        "offset=2147483648",
        "limit=bad",
    ] {
        let request = test::TestRequest::get()
            .uri(&format!("/projects?{query}"))
            .insert_header(("Authorization", bearer(alice, false)))
            .to_request();
        assert_eq!(
            test::call_service(&app, request).await.status(),
            StatusCode::BAD_REQUEST
        );
    }
    for fields in [
        json!({}),
        json!({"project_id":"bad"}),
        json!({"project_id":4}),
    ] {
        let request = test::TestRequest::put()
            .uri(&format!("/chats/{}/project", chat.id))
            .insert_header(("Authorization", bearer(alice, false)))
            .set_json(fields)
            .to_request();
        assert_eq!(
            test::call_service(&app, request).await.status(),
            StatusCode::BAD_REQUEST
        );
    }
    let request = test::TestRequest::get()
        .uri("/chats")
        .insert_header(("Authorization", bearer(alice, false)))
        .to_request();
    let chats: Value = test::call_and_read_body_json(&app, request).await;
    assert_eq!(chats["chats"][0]["chat_id"], json!(chat.id));
    assert!(chats["chats"][0]["project_id"].is_null());
}

/// Check real HTTP forwarding, context-floor parity, continuation matching and untrusted-field stripping.
#[sqlx::test(migrations = "../db/migrations")]
async fn all_rounds_receive_only_owned_stored_project_context(pool: PgPool) {
    let alice = owner(&pool).await;
    let fixture: Value =
        serde_json::from_str(include_str!("../../../../contracts/project-memory.json")).unwrap();
    let expected = &fixture["request"];
    let scope = db::project::create(
        &pool,
        alice,
        expected["project"]["name"].as_str().unwrap(),
        expected["project"]["description"].as_str().unwrap(),
        expected["project"]["notes"].as_str().unwrap(),
    )
    .await
    .unwrap();
    let source = db::conversation::create_in_project(&pool, alice, "source", Some(scope.id))
        .await
        .unwrap();
    let chat = db::conversation::create_in_project(&pool, alice, "current", Some(scope.id))
        .await
        .unwrap();
    let memory = db::conversation::append(
        &pool,
        alice,
        source.id,
        &message(
            "assistant",
            expected["memories"][0]["memory"].as_str().unwrap(),
        ),
        &[],
    )
    .await
    .unwrap();
    let logs: Requests = Arc::new(Mutex::new(Vec::new()));
    let engine = engine(logs.clone());
    let proxy = Proxy::new(engine.url("").trim_end_matches('/').to_owned(), None);
    let app = test::init_service(
        App::new()
            .app_data(web::Data::new(pool.clone()))
            .app_data(web::Data::new(SECRET.to_owned()))
            .app_data(web::Data::new(proxy.clone()))
            .configure(api::configure),
    )
    .await;
    let question = format!(
        "{} assistant",
        expected["turns"][0]["text"].as_str().unwrap()
    );
    let request = test::TestRequest::post()
        .uri(&format!("/chats/{}/messages", chat.id))
        .insert_header(("Authorization", bearer(alice, false)))
        .set_json(
            json!({"model_name":"tiny","text":question,"project":{"notes":"forged"},
            "project_id":Uuid::new_v4(),"memories":[{"memory":"forged"}]}),
        )
        .to_request();
    assert_eq!(
        test::call_service(&app, request).await.status(),
        StatusCode::CREATED
    );
    let next = api::conversation::prepare(&pool, &proxy, alice, chat.id,
        json!({"model_name":"tiny","tool_result":{"body":"other unrelated terms"},"project":{},"memories":[{}]})).await.unwrap();
    assert_eq!(next.config["memories"][0]["message"], json!(memory.id));
    let steering = api::conversation::prepare(
        &pool,
        &proxy,
        alice,
        chat.id,
        json!({"model_name":"tiny","tool_result":{},"interject":"unmatchedword"}),
    )
    .await
    .unwrap();
    assert_eq!(steering.config["memories"], json!([]));
    let floor_request = test::TestRequest::get()
        .uri(&format!("/chats/{}/context-floor?model_name=tiny", chat.id))
        .insert_header(("Authorization", bearer(alice, false)))
        .to_request();
    assert_eq!(
        test::call_service(&app, floor_request).await.status(),
        StatusCode::OK
    );
    let standalone = db::conversation::create(&pool, alice, "standalone")
        .await
        .unwrap();
    let ordinary = api::conversation::prepare(&pool, &proxy, alice, standalone.id,
        json!({"model_name":"tiny","text":question,"project":expected["project"],"memories":expected["memories"]})).await.unwrap();
    assert!(ordinary.config.get("project").is_none());
    assert!(ordinary.config.get("memories").is_none());
    let floor = api::conversation::context_request(
        &pool,
        alice,
        chat.id,
        &[],
        json!({"model_name":"tiny","turns":[],"project":{},"memories":[{}]}),
    )
    .await
    .unwrap();
    assert_eq!(floor["project"]["project_id"], json!(scope.id));
    assert_eq!(floor["memories"], json!([]));
    let recorded = logs.lock().unwrap();
    let preparations = recorded
        .iter()
        .filter(|(operation, _)| operation == "prepare")
        .collect::<Vec<_>>();
    for (_, body) in &preparations {
        assert!(body.get("project_id").is_none());
    }
    assert_eq!(
        preparations[0].1["project"]["notes"],
        expected["project"]["notes"]
    );
    assert_eq!(
        preparations[0].1["memories"][0]["message"],
        json!(memory.id)
    );
    assert!(preparations.last().unwrap().1.get("project").is_none());
    assert!(preparations.last().unwrap().1.get("memories").is_none());
    let config = &recorded
        .iter()
        .find(|(operation, _)| operation == "generate")
        .unwrap()
        .1;
    assert_eq!(config["project"]["name"], expected["project"]["name"]);
    assert_eq!(
        config["project"]["description"],
        expected["project"]["description"]
    );
    assert_eq!(config["project"]["notes"], expected["project"]["notes"]);
    assert_eq!(config["project"]["project_id"], json!(scope.id));
    assert_eq!(
        config["memories"][0]["memory"],
        expected["memories"][0]["memory"]
    );
    assert_eq!(config["memories"][0]["thread"], json!(source.id));
    assert_eq!(config["memories"][0]["message"], json!(memory.id));
    assert_eq!(config["memories"][0]["role"], "assistant");
    assert!(config["memories"][0]["age_seconds"].as_u64().is_some());
    assert_eq!(
        recorded
            .iter()
            .find(|(operation, _)| operation == "floor")
            .unwrap()
            .1["project"],
        config["project"]
    );
}

/// Preparation rejection leaves accepted storage untouched, and a history floor cannot recall a hidden user query.
#[sqlx::test(migrations = "../db/migrations")]
async fn failed_preparation_and_visible_history_limit_retrieval(pool: PgPool) {
    let alice = owner(&pool).await;
    let scope = db::project::create(&pool, alice, "orbit", "", "metric units")
        .await
        .unwrap();
    let source = db::conversation::create_in_project(&pool, alice, "source", Some(scope.id))
        .await
        .unwrap();
    let chat = db::conversation::create_in_project(&pool, alice, "current", Some(scope.id))
        .await
        .unwrap();
    db::conversation::append(
        &pool,
        alice,
        source.id,
        &message("user", "orbit battery"),
        &[],
    )
    .await
    .unwrap();
    let logs: Requests = Arc::new(Mutex::new(Vec::new()));
    let engine = engine(logs.clone());
    let proxy = Proxy::new(engine.url("").trim_end_matches('/').to_owned(), None);
    let rejected = api::conversation::prepare(
        &pool,
        &proxy,
        alice,
        chat.id,
        json!({"model_name":"incompatible","text":"orbit battery"}),
    )
    .await;
    assert!(matches!(
        rejected,
        Err(service::Error::Status { code: 400, .. })
    ));
    assert!(db::conversation::messages(&pool, alice, chat.id)
        .await
        .unwrap()
        .is_empty());
    assert_eq!(
        logs.lock().unwrap()[0].1["project"]["notes"],
        "metric units"
    );
    api::conversation::prepare(
        &pool,
        &proxy,
        alice,
        chat.id,
        json!({"model_name":"tiny","text":"orbit battery"}),
    )
    .await
    .unwrap();
    let floor =
        db::conversation::append(&pool, alice, chat.id, &message("assistant", "answer"), &[])
            .await
            .unwrap();
    let continued = api::conversation::prepare(
        &pool,
        &proxy,
        alice,
        chat.id,
        json!({"model_name":"tiny","tool_result":{},"history_floor":floor.id}),
    )
    .await
    .unwrap();
    assert_eq!(continued.config["memories"], json!([]));
    assert_eq!(continued.config["project"]["notes"], "metric units");
}

/// Project assignment waits for an active turn's lease and reloads the new project after completion.
#[sqlx::test(migrations = "../db/migrations")]
async fn assignment_waits_until_the_active_turn_finishes(pool: PgPool) {
    tokio::task::LocalSet::new()
        .run_until(async move {
            let alice = owner(&pool).await;
            let first = db::project::create(&pool, alice, "first", "", "first instructions")
                .await
                .unwrap();
            let second = db::project::create(&pool, alice, "second", "", "second instructions")
                .await
                .unwrap();
            let chat = db::conversation::create_in_project(&pool, alice, "current", Some(first.id))
                .await
                .unwrap();
            let lease = api::conversation::lock(chat.id).await;
            let server_pool = pool.clone();
            let backend = actix_test::start(move || {
                App::new()
                    .app_data(web::Data::new(server_pool.clone()))
                    .app_data(web::Data::new(SECRET.to_owned()))
                    .configure(api::configure)
            });
            let client = awc::Client::default();
            let mut pending = Box::pin(
                client
                    .put(backend.url(&format!("/chats/{}/project", chat.id)))
                    .insert_header(("Authorization", bearer(alice, false)))
                    .send_json(&json!({"project_id":second.id})),
            );
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(100), &mut pending)
                    .await
                    .is_err()
            );
            assert_eq!(
                db::conversation::get(&pool, alice, chat.id)
                    .await
                    .unwrap()
                    .unwrap()
                    .project_id,
                Some(first.id)
            );
            drop(lease);
            assert_eq!(
                tokio::time::timeout(std::time::Duration::from_secs(2), pending)
                    .await
                    .unwrap()
                    .unwrap()
                    .status(),
                StatusCode::OK
            );
            let context = db::project::context(&pool, alice, chat.id, None)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(context.project.notes, "second instructions");
        })
        .await;
}

/// The real browser websocket path uses the same owned stored-context assembly as HTTP generation.
#[sqlx::test(migrations = "../db/migrations")]
async fn websocket_generation_uses_stored_context(pool: PgPool) {
    tokio::task::LocalSet::new()
        .run_until(async move {
            let alice = owner(&pool).await;
            let scope = db::project::create(&pool, alice, "orbit", "", "metric units")
                .await
                .unwrap();
            let source =
                db::conversation::create_in_project(&pool, alice, "source", Some(scope.id))
                    .await
                    .unwrap();
            let chat = db::conversation::create_in_project(&pool, alice, "current", Some(scope.id))
                .await
                .unwrap();
            let memory = db::conversation::append(
                &pool,
                alice,
                source.id,
                &message("user", "orbit battery"),
                &[],
            )
            .await
            .unwrap();
            let logs: Requests = Arc::new(Mutex::new(Vec::new()));
            let engine = engine(logs.clone());
            let proxy = Proxy::new(engine.url("").trim_end_matches('/').to_owned(), None);
            let mut backend = actix_test::start(move || {
                App::new()
                    .app_data(web::Data::new(pool.clone()))
                    .app_data(web::Data::new(SECRET.to_owned()))
                    .app_data(web::Data::new(proxy.clone()))
                    .app_data(web::Data::new(tools::Registry::new(vec![])))
                    .configure(api::configure)
            });
            let mut socket = backend
                .ws_at(&format!("/ws/chats/{}", chat.id))
                .await
                .unwrap();
            let token = bearer(alice, false)
                .trim_start_matches("Bearer ")
                .to_owned();
            socket
                .send(actix_http::ws::Message::Text(
                    json!({"token":token}).to_string().into(),
                ))
                .await
                .unwrap();
            let mut done = false;
            while let Some(Ok(frame)) = socket.next().await {
                let actix_http::ws::Frame::Text(text) = frame else {
                    continue;
                };
                let frame: Value = serde_json::from_slice(&text).unwrap();
                if frame["type"] == "ready" {
                    socket
                        .send(actix_http::ws::Message::Text(
                            json!({"model_name":"tiny","text":"orbit schedule?",
                    "project":{"notes":"forged"},"memories":[{}]})
                            .to_string()
                            .into(),
                        ))
                        .await
                        .unwrap();
                }
                if frame["type"] == "done" {
                    done = true;
                }
                assert_ne!(frame["type"], "error", "{frame}");
            }
            assert!(done);
            let recorded = logs.lock().unwrap();
            let config = &recorded
                .iter()
                .find(|(operation, _)| operation == "stream")
                .unwrap()
                .1;
            assert_eq!(config["project"]["notes"], "metric units");
            assert_eq!(config["memories"][0]["message"], json!(memory.id));
        })
        .await;
}

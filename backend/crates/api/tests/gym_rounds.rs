//! A gym run, over the socket the run page actually opens.
//!
//! Two stubs: an engine playing a fixed SSE script per round, and a gyms
//! service answering JSON. What is under test is the driver between them —
//! that a `tool_call` becomes a step in the episode's own workspace, that the
//! output comes back rendered the way chat renders it, that the episode is
//! graded when the model stops calling, and that a model which never stops is
//! stopped and graded anyway rather than left unscored.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc::{Receiver, channel};

use actix_web::{App, web};
use auth::AccessLevel;
use auth::token;
use futures_util::{SinkExt, StreamExt};
use llm_engine::Proxy;
use serde_json::{Value, json};
use tools::Registry;
use uuid::Uuid;

const JWT_SECRET: &str = "gym-rounds-test-jwt-secret";
const SERVICE_SECRET: &str = "gym-rounds-test-service-secret";

const RUN_ID: &str = "11111111-1111-4111-8111-111111111111";
const EPISODE_ID: &str = "22222222-2222-4222-8222-222222222222";

const CALLS_THE_SHELL: &str = concat!(
    "event: token\ndata: {\"token\": \"<|tool-call|>\"}\n\n",
    "event: tool_call\ndata: {\"tool\": \"bash\", \"argument\": \"grep -rl NEEDLE .\"}\n\n",
    "event: done\ndata: {\"text\": \"<|tool-call|>{}<|/tool-call|>\", \"usage\": {\"elapsed_secs\": 1}}\n\n",
);

const ANSWERS_PLAINLY: &str = concat!(
    "event: token\ndata: {\"token\": \"report.txt\"}\n\n",
    "event: done\ndata: {\"text\": \"<|par|>report.txt<|/par|>\", \"usage\": {\"elapsed_secs\": 1}}\n\n",
);

/// The LLM engine: `script[n]` to the nth round, the last one repeating.
fn engine_stub(script: Vec<&'static str>) -> (String, Receiver<String>) {
    serve(move |nth, _request| {
        let body = script[nth.min(script.len() - 1)];
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    })
}

/// The gyms engine: one run, one episode, and the two writes a driver makes.
fn gyms_stub() -> (String, Receiver<String>) {
    gym_with_tools(json!(["bash"]))
}

fn gym_with_tools(available: Value) -> (String, Receiver<String>) {
    gym_with_episodes(json!([{
        "id": EPISODE_ID, "index": 0, "task_type": "bash_grep", "seed": 7,
        "status": "pending", "prompt": "Which file holds NEEDLE?",
        "system": "You are working at a Linux shell.", "tools": available,
    }]))
}

fn gym_with_episodes(episodes: Value) -> (String, Receiver<String>) {
    serve(move |_nth, request| {
        let body = if request.contains(&format!("GET /gym/runs/{RUN_ID} ")) {
            json!({
                "id": RUN_ID,
                "model_name": "tiny",
                "episode_count": 1,
                "score": 1.0,
                "correct": 1,
                "graded": 1,
                "episodes": episodes,
            })
        } else if request.contains("/step") {
            json!({ "stdout": "report.txt", "stderr": "", "exit_code": 0, "timed_out": false,
                    "step_num": 0, "reward": 0.08, "reward_total": 0.08 })
        } else if request.contains("/finish") {
            json!({ "episode_id": EPISODE_ID, "score": 1.0, "correct": true,
                    "answer": "report.txt", "ground_truth": "report.txt" })
        } else {
            json!({ "ok": true })
        }
        .to_string();
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    })
}

/// A one-connection-at-a-time HTTP stub that records every request it answered.
fn serve(answer: impl Fn(usize, &str) -> String + Send + 'static) -> (String, Receiver<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let url = format!("http://{}", listener.local_addr().expect("addr"));
    let (tx, rx) = channel();

    std::thread::spawn(move || {
        for (nth, socket) in listener.incoming().enumerate() {
            let Ok(mut socket) = socket else { return };
            let mut buffer = [0u8; 8192];
            let read = socket.read(&mut buffer).unwrap_or(0);
            let request = String::from_utf8_lossy(&buffer[..read]).to_string();

            let _ = socket.write_all(answer(nth, &request).as_bytes());
            let _ = socket.flush();
            if tx.send(request).is_err() {
                return;
            }
        }
    });

    (url, rx)
}

fn server(engine_url: String, gyms_url: String) -> actix_test::TestServer {
    actix_test::start(move || {
        App::new()
            .app_data(web::Data::new(JWT_SECRET.to_string()))
            .app_data(web::Data::new(Proxy::new(
                engine_url.clone(),
                Some(SERVICE_SECRET.to_string()),
            )))
            .app_data(web::Data::new(gyms_engine::Proxy::new(
                gyms_url.clone(),
                Some(SERVICE_SECRET.to_string()),
            )))
            .app_data(web::Data::new(Registry::new(vec![])))
            .configure(api::configure)
    })
}

/// Admin, because every gym route is: a run names a model and spends the GPU.
fn admin_token() -> String {
    let jwt =
        token::issue_user_token(Uuid::new_v4(), AccessLevel::Admin, JWT_SECRET).expect("issued");
    auth::token_cipher::encrypt(&jwt, JWT_SECRET).expect("sealed")
}

async fn drive(server: &mut actix_test::TestServer) -> Vec<Value> {
    let mut socket = server
        .ws_at(&format!("/ws/gyms/runs/{RUN_ID}"))
        .await
        .expect("socket opened");

    socket
        .send(actix_http::ws::Message::Text(
            json!({ "token": admin_token() }).to_string().into(),
        ))
        .await
        .expect("auth frame sent");

    let mut frames = Vec::new();
    let mut started = false;
    while let Some(Ok(message)) = socket.next().await {
        let actix_http::ws::Frame::Text(bytes) = message else {
            continue;
        };
        let frame: Value = serde_json::from_slice(&bytes).expect("json frame");

        if frame["type"] == "ready" && !started {
            started = true;
            socket
                .send(actix_http::ws::Message::Text(
                    json!({ "temperature": 0.7 }).to_string().into(),
                ))
                .await
                .expect("settings sent");
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

fn requests(rx: &Receiver<String>, n: usize) -> Vec<String> {
    (0..n).map(|_| rx.recv().expect("a request")).collect()
}

#[actix_web::test]
async fn a_tool_free_episode_never_dispatches_even_an_unexpected_tool_call() {
    let (engine_url, asked) = engine_stub(vec![CALLS_THE_SHELL]);
    let (gyms_url, served) = gym_with_tools(json!([]));
    let mut server = server(engine_url, gyms_url);

    let frames = drive(&mut server).await;
    let request = body_of(&asked.recv().expect("one generation"));
    assert_eq!(request["max_tool_calls"], 0);
    assert!(asked.try_recv().is_err());
    assert!(!kinds(&frames).contains(&"tool_result"));
    assert!(kinds(&frames).contains(&"episode_done"));
    assert!(served.try_iter().all(|request| !request.contains("/step")));
}

#[actix_web::test]
async fn an_episode_runs_its_command_and_is_graded_on_the_reply_that_stops_calling() {
    // Both receivers stay bound: dropping one ends the stub thread it drains,
    // mid-run.
    let (engine_url, _asked) = engine_stub(vec![CALLS_THE_SHELL, ANSWERS_PLAINLY]);
    let (gyms_url, _served) = gyms_stub();
    let mut server = server(engine_url, gyms_url);

    let frames = drive(&mut server).await;

    // The LLM engine's `done` arrives as `round_done`: an episode takes a round per
    // command, and a client counting `done` would end it every time.
    assert_eq!(
        kinds(&frames),
        vec![
            "episode_start",
            "token",
            "tool_call",
            "round_done",
            "tool_result",
            "token",
            "round_done",
            "episode_done",
            "run_done",
        ]
    );

    // Every frame says which episode it belongs to, so one socket carries a
    // whole run.
    for frame in &frames {
        if frame["type"] != "run_done" {
            assert_eq!(frame["episode_id"], EPISODE_ID, "{frame}");
        }
    }

    let verdict = frames
        .iter()
        .find(|f| f["type"] == "episode_done")
        .expect("graded");
    assert_eq!(verdict["score"], 1.0);
    assert_eq!(verdict["ground_truth"], "report.txt");

    // What the model reads is what a chat round would have shown it: the output
    // inside a code block, rendered by the tool's own renderer.
    let result = frames
        .iter()
        .find(|f| f["type"] == "tool_result")
        .expect("a result");
    assert_eq!(result["body"], "<|codeblock|>report.txt<|/codeblock|>");
}

#[actix_web::test]
async fn the_second_round_carries_the_whole_conversation_with_the_call_named() {
    let (engine_url, asked) = engine_stub(vec![CALLS_THE_SHELL, ANSWERS_PLAINLY]);
    let (gyms_url, _served) = gyms_stub();
    let mut server = server(engine_url, gyms_url);

    drive(&mut server).await;

    // Stateless generation: an episode is not a conversation anybody should
    // find in their chat list, so the whole thing is sent every round.
    let first = body_of(&asked.recv().expect("a first round"));
    assert_eq!(first["turns"][0]["role"], "user");
    assert_eq!(first["turns"][0]["text"], "Which file holds NEEDLE?");
    assert_eq!(first["system"], "You are working at a Linux shell.");
    assert_eq!(first["temperature"], 0.7);

    let second = body_of(&asked.recv().expect("a second round"));
    assert_eq!(second["turns"][1]["role"], "assistant");
    // The tool turn names its call rather than spelling the block: the LLM engine
    // renders it, so there is one copy of that layout and not two.
    assert_eq!(second["turns"][2]["role"], "tool");
    assert_eq!(second["turns"][2]["tool"], "bash");
    assert_eq!(second["turns"][2]["argument"], "grep -rl NEEDLE .");
    assert_eq!(
        second["turns"][2]["text"],
        "<|codeblock|>report.txt<|/codeblock|>"
    );
}

#[actix_web::test]
async fn a_command_runs_in_the_episodes_own_workspace() {
    let (engine_url, _asked) = engine_stub(vec![CALLS_THE_SHELL, ANSWERS_PLAINLY]);
    let (gyms_url, served) = gyms_stub();
    let mut server = server(engine_url, gyms_url);

    drive(&mut server).await;

    // run, status(running), step, finish, status(done), run — the step is keyed
    // by the episode, which is the sandbox session key.
    let seen = requests(&served, 4);
    let step = seen.iter().find(|r| r.contains("/step")).expect("a step");
    assert!(
        step.contains(&format!("/gym/episodes/{EPISODE_ID}/step")),
        "{step}"
    );
    assert_eq!(body_of(step)["command"], "grep -rl NEEDLE .");
}

#[actix_web::test]
async fn a_model_that_never_stops_calling_is_stopped_and_graded_anyway() {
    // An unscored episode would let a model that never stops looking around be
    // judged on the one task it happened to finish.
    unsafe { std::env::set_var("MAX_GYM_ROUNDS", "2") };
    let (engine_url, _asked) = engine_stub(vec![CALLS_THE_SHELL]);
    let (gyms_url, served) = gyms_stub();
    let mut server = server(engine_url, gyms_url);

    let frames = drive(&mut server).await;
    unsafe { std::env::remove_var("MAX_GYM_ROUNDS") };

    assert_eq!(kinds(&frames).last(), Some(&"run_done"));
    assert!(frames.iter().any(|f| f["type"] == "episode_done"));
    assert!(requests(&served, 5).iter().any(|r| r.contains("/finish")));
}

#[actix_web::test]
async fn a_general_run_uses_each_episodes_own_tool_policy() {
    let (engine_url, asked) = engine_stub(vec![ANSWERS_PLAINLY, CALLS_THE_SHELL, ANSWERS_PLAINLY]);
    let (gyms_url, served) = gym_with_episodes(json!([
        {"id": "math-episode", "index": 0, "gym_id": "math", "task_type": "math_addition_easy",
         "status": "pending", "prompt": "1+1?", "system": "Answer.", "tools": []},
        {"id": EPISODE_ID, "index": 1, "gym_id": "bash", "task_type": "bash_grep",
         "status": "pending", "prompt": "Find the file.", "system": "Use bash.", "tools": ["bash"]}
    ]));
    let mut server = server(engine_url, gyms_url);
    let frames = drive(&mut server).await;
    let generations = requests(&asked, 3);
    assert_eq!(body_of(&generations[0])["max_tool_calls"], 0);
    assert_ne!(body_of(&generations[1])["max_tool_calls"], 0);
    let steps: Vec<_> = served.try_iter().filter(|r| r.contains("/step")).collect();
    assert_eq!(steps.len(), 1);
    assert!(steps[0].contains(EPISODE_ID));
    assert_eq!(
        frames
            .iter()
            .filter(|f| f["type"] == "episode_done")
            .count(),
        2
    );
}

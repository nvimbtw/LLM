//! One gym run: a model against every task in it, driven here.
//!
//! A gym episode is the same shape as a chat turn — generate, run what the
//! reply asked for, generate again with the answer in the conversation — and it
//! is driven from this side for the same reason chat is: the sandbox's answer
//! has to be *rendered* into the marker vocabulary before the model reads it,
//! and `tools::observation` is that rendering. A driver anywhere else would be
//! a second copy of it, and a model trained against a second copy has been
//! trained on a layout serving never shows. See
//! `docs/decisions/gyms-are-driven-by-the-backend.md`.
//!
//! What is different from `turn.rs`, and why this is not that function with a
//! flag: a chat turn is stored by the engine and comes back as history, so each
//! round sends one `tool_result` and nothing else. An episode has no chat — the
//! conversation is held here, in `turns`, and handed whole to the engine's
//! stateless `/chat/stream` every round. The two loops share their frame
//! vocabulary (`turn::envelope`) and their tool rendering, which is the part
//! that must not drift; the rest is genuinely two different jobs.

use engine::Frame;
use futures_util::StreamExt;
use serde_json::{Value, json};
use uuid::Uuid;

use crate::{turn, ws};

/// How many times an episode may be handed command output and asked to carry
/// on. Higher than chat's six because nobody is waiting on it and the tasks
/// take real looking-around: a `ls`, a `grep` that misses, a `grep` that lands.
const DEFAULT_ROUNDS: usize = 8;

/// The reply budget for one round. A gym answer is a filename or a short token,
/// so this is spent on the commands and the reasoning around them.
const MAX_TOKENS: u32 = 256;

/// The only tool this gym implements. Named here rather than matched inline so
/// a second gym's tool is a row beside it and not a branch inside `play`.
const BASH: &str = tools::Bash::NAME;

fn max_rounds() -> usize {
    std::env::var("MAX_GYM_ROUNDS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(DEFAULT_ROUNDS)
}

/// Drive every episode of `run_id`, relaying everything to `session` as it
/// happens.
pub async fn run(
    session: &mut actix_ws::Session,
    engine: &engine::Proxy,
    gyms: &gyms::Proxy,
    caller: &auth::Caller,
    run_id: Uuid,
    ask: &Value,
) {
    let record = match gyms.get(caller, &format!("/gym/runs/{run_id}")).await {
        Ok(record) => record,
        Err(e) => {
            let _ = session.text(ws::error_frame(&e.to_string())).await;
            return;
        }
    };

    let model = turn::string_at(&record, "model_name");
    let budget = tools::Config::from_env().bash_budget_chars;
    set_status(gyms, caller, run_id, "running").await;

    let episodes = record["episodes"].as_array().cloned().unwrap_or_default();
    for episode in &episodes {
        if !matches!(episode_status(episode), "pending" | "running") {
            continue;
        }
        if !start(session, episode).await {
            set_status(gyms, caller, run_id, "abandoned").await;
            return;
        }
        match episode_loop(session, engine, gyms, caller, episode, &model, ask, budget).await {
            Some(()) => {}
            // The browser hung up mid-episode. Whatever finished is graded and
            // stored; the rest of the run is not run behind a closed socket.
            None => {
                close(gyms, caller, episode).await;
                set_status(gyms, caller, run_id, "abandoned").await;
                return;
            }
        }
    }

    set_status(gyms, caller, run_id, "done").await;
    if let Ok(finished) = gyms.get(caller, &format!("/gym/runs/{run_id}")).await {
        let _ = session
            .text(
                turn::envelope(
                    "run_done",
                    &json!({
                        "score": finished["score"].clone(),
                        "correct": finished["correct"].clone(),
                        "graded": finished["graded"].clone(),
                        "episode_count": finished["episode_count"].clone(),
                    }),
                )
                .to_string(),
            )
            .await;
    }
}

/// One episode, round by round. None when the browser hung up.
///
/// The reply that stops calling the shell is the answer, and so is the reply of
/// the **terminal round** — the last one, generated with the tool-call marker
/// masked out of the grammar so the model cannot reach for the shell again.
/// Masked rather than asked: a model this size does not reliably follow an
/// instruction to stop, and the first measured run ended 22 of 30 episodes
/// still calling. Those graded as zero, correctly, but a zero for never
/// answering says nothing about whether the model had found the answer — and a
/// group of rollouts that all score zero for that reason is a group no policy
/// gradient can learn from.
///
/// An episode absent from the run's mean would let a model that never stops
/// calling look like the one lucky episode it did finish, which is why the
/// terminal round grades whatever it says rather than closing the episode.
#[allow(clippy::too_many_arguments)]
async fn episode_loop(
    session: &mut actix_ws::Session,
    engine: &engine::Proxy,
    gyms: &gyms::Proxy,
    caller: &auth::Caller,
    episode: &Value,
    model: &str,
    ask: &Value,
    budget: usize,
) -> Option<()> {
    let episode_id = turn::string_at(episode, "id");
    let tag = json!({ "episode_id": episode_id });
    let mut turns = vec![json!({
        "role": "user",
        "text": turn::string_at(episode, "prompt"),
    })];

    let rounds = max_rounds();
    for round in 0..rounds {
        // The last round is the terminal one: generated with the tool-call
        // marker masked out, so the model has to say something rather than
        // reach for the shell again. Without it 22 of 30 episodes in the first
        // measured run ended still calling — graded, correctly, as zero, but
        // measuring nothing about whether the model had found the answer.
        let last_round = round + 1 == rounds;
        let body = generate(model, episode, &turns, ask, last_round.then_some(0));
        let frames = match engine.stream(caller, "/chat/stream", body).await {
            Ok(frames) => frames,
            Err(e) => {
                session.text(ws::error_frame(&e.to_string())).await.ok()?;
                return Some(());
            }
        };

        let reply = relay(session, frames, &tag).await?;
        // On the terminal round `calls` is empty by construction; the check is
        // kept so a build whose engine predates `max_tool_calls` still ends.
        if reply.calls.is_empty() || last_round {
            let graded = finish(gyms, caller, &episode_id, &reply.text).await;
            session.text(turn::envelope("episode_done", &graded).to_string()).await.ok()?;
            return Some(());
        }

        turns.push(json!({ "role": "assistant", "text": reply.text }));
        let answered = step_through(session, gyms, caller, &episode_id, &reply, budget).await?;
        turns.push(answered);
    }
    Some(())
}

/// Run everything the round asked for, in order, and return the turn that
/// answers the first call.
///
/// In order because the calls share one filesystem: a round that writes a file
/// and then reads it is a round whose second command depends on the first. Only
/// the first answer continues the episode, which is the rule the engine sets for
/// a chat turn and is kept here so an episode's conversation has the shape
/// serving produces. The rest are still executed, recorded and reported — a
/// command the model ran is a step whether or not it read the output.
async fn step_through(
    session: &mut actix_ws::Session,
    gyms: &gyms::Proxy,
    caller: &auth::Caller,
    episode_id: &str,
    reply: &Reply,
    budget: usize,
) -> Option<Value> {
    let mut answer: Option<Value> = None;
    for call in &reply.calls {
        let argument = turn::string_at(call, "argument");
        let tool = turn::string_at(call, "tool");
        let body = execute(gyms, caller, episode_id, &tool, &argument, &reply.text, budget).await;
        let result = json!({
            "tool": tool,
            "argument": argument,
            "body": body,
            "episode_id": episode_id,
        });
        session.text(turn::envelope("tool_result", &result).to_string()).await.ok()?;
        answer.get_or_insert(json!({
            "role": "tool",
            "tool": result["tool"].clone(),
            "argument": result["argument"].clone(),
            "text": result["body"].clone(),
        }));
    }
    answer
}

/// One tool call, run in the episode's workspace and recorded, with what the
/// model reads back rendered under `observation`.
///
/// **Shared by both drivers**, and the rendering is the whole reason it is
/// shared: the browser's socket loop below and the RL rollout in the engine
/// reach this through `routes::internal`, and `tools::observation` is the same
/// function the live chat `bash` tool renders with. A gym whose observations
/// are a near-copy of the real ones trains a model for a shell it is never
/// served — see `docs/decisions/gyms-are-driven-by-the-backend.md`.
///
/// The gyms service owns the episode's filesystem, its step record and its
/// reward, so this cannot go through `Tool::run`: that reaches the *live*
/// sandbox. What is shared with serving is the layout, and `render` below is
/// where a second gym names its own.
pub async fn play(
    gyms: &gyms::Proxy,
    caller: &auth::Caller,
    episode_id: &str,
    tool: &str,
    argument: &str,
    reply: &str,
    budget: usize,
) -> Result<Value, gyms::GymsError> {
    if tool != BASH {
        // A call to something this gym does not have is not a failed episode:
        // the reply that made it is already generated, and the model reading
        // "that did not work" is a far better round than a run that stops. It
        // costs no step and earns no reward, because nothing ran.
        let caveat = format!("There is no {tool} tool here. This gym has a shell.");
        return Ok(json!({ "observation": tools::markup::warning(&caveat) }));
    }

    let path = format!("/gym/episodes/{episode_id}/step");
    let mut ran = gyms.post(caller, &path, json!({ "command": argument, "reply": reply })).await?;
    let observation = tools::observation(
        tools::Output {
            stdout: turn::string_at(&ran, "stdout"),
            stderr: turn::string_at(&ran, "stderr"),
            exit_code: ran["exit_code"].as_i64().unwrap_or_default() as i32,
            timed_out: ran["timed_out"].as_bool().unwrap_or_default(),
        },
        budget,
    );
    merge(&mut ran, &json!({ "observation": observation }));
    Ok(ran)
}

/// One command, run in the episode's workspace and rendered for the model.
///
/// A shell the gyms service could not reach comes back as a caveat rather than
/// as a failed episode, for the same reason `render_empty` does.
async fn execute(
    gyms: &gyms::Proxy,
    caller: &auth::Caller,
    episode_id: &str,
    tool: &str,
    command: &str,
    reply: &str,
    budget: usize,
) -> String {
    match play(gyms, caller, episode_id, tool, command, reply, budget).await {
        Ok(ran) => turn::string_at(&ran, "observation"),
        Err(e) => tools::markup::warning(&format!("The shell could not be reached: {e}")),
    }
}

/// What one round of generation produced.
struct Reply {
    text: String,
    calls: Vec<Value>,
}

/// Forward a round's frames, tagged with the episode they belong to.
///
/// None when the browser hung up. The engine's own `done` is relayed as
/// `round_done`: a gym client counts episodes, and a `done` per round would
/// look like an episode ending every time the model ran a command.
async fn relay(
    session: &mut actix_ws::Session,
    frames: impl futures_util::Stream<Item = engine::Result<Frame>>,
    tag: &Value,
) -> Option<Reply> {
    futures_util::pin_mut!(frames);

    let mut calls = Vec::new();
    let mut text = String::new();
    while let Some(frame) = frames.next().await {
        let message = match frame {
            Ok(frame) => {
                if frame.kind == "tool_call" {
                    calls.push(frame.data.clone());
                }
                if frame.kind == "done" {
                    text = turn::string_at(&frame.data, "text");
                }
                let kind = if frame.kind == "done" { "round_done" } else { &frame.kind };
                let mut payload = turn::envelope(kind, &frame.data);
                merge(&mut payload, tag);
                payload.to_string()
            }
            Err(e) => ws::error_frame(&e.to_string()),
        };
        session.text(message).await.ok()?;
    }
    Some(Reply { text, calls })
}

/// The request that generates one round.
///
/// `turns` is the whole conversation every time, because `/chat/stream` stores
/// nothing — which is the property that makes it the right route for this: an
/// episode is not a conversation anybody should be able to find in their chat
/// list afterwards.
/// `tool_calls` is the ceiling for this round: None leaves the engine's own
/// per-turn budget alone, `Some(0)` masks the tool-call marker out of the
/// grammar so the round can only answer in prose. See `terminal`.
fn generate(
    model: &str,
    episode: &Value,
    turns: &[Value],
    ask: &Value,
    tool_calls: Option<u32>,
) -> Value {
    let mut body = json!({
        "model_name": model,
        "system": turn::string_at(episode, "system"),
        "turns": turns,
        "max_tokens": MAX_TOKENS,
    });
    if let Some(limit) = tool_calls {
        body["max_tool_calls"] = json!(limit);
    }
    // Sampling the client chose, if it chose any. The model is the run's and
    // not the socket's — it is what the run was created against and what its
    // score will be recorded under.
    //
    // `context_multiplier` is carried for a reason particular to this: a gym
    // task plus its system block is ~70 tokens before the shell has answered
    // anything, and `build_prompt` reserves half the window for the reply — so
    // a model trained under 256 cannot hold a task at all, and stretching the
    // window is the only way to run one against it.
    for key in [
        "temperature", "top_k", "top_p", "max_tokens", "budget", "seed", "context_multiplier",
    ] {
        if let Some(value) = ask.get(key) {
            body[key] = value.clone();
        }
    }
    body
}

async fn finish(gyms: &gyms::Proxy, caller: &auth::Caller, episode_id: &str, reply: &str) -> Value {
    let path = format!("/gym/episodes/{episode_id}/finish");
    match gyms.post(caller, &path, json!({ "reply": reply })).await {
        Ok(graded) => graded,
        Err(e) => json!({ "episode_id": episode_id, "error": e.to_string() }),
    }
}

async fn close(gyms: &gyms::Proxy, caller: &auth::Caller, episode: &Value) {
    let id = turn::string_at(episode, "id");
    let _ = gyms.post(caller, &format!("/gym/episodes/{id}/close"), Value::Null).await;
}

async fn set_status(gyms: &gyms::Proxy, caller: &auth::Caller, run_id: Uuid, status: &str) {
    let path = format!("/gym/runs/{run_id}/status");
    let _ = gyms.post(caller, &path, json!({ "status": status })).await;
}

async fn start(session: &mut actix_ws::Session, episode: &Value) -> bool {
    let frame = turn::envelope(
        "episode_start",
        &json!({
            "episode_id": turn::string_at(episode, "id"),
            "index": episode["index"].clone(),
            "task_type": turn::string_at(episode, "task_type"),
            "seed": episode["seed"].clone(),
            "prompt": turn::string_at(episode, "prompt"),
            "system": turn::string_at(episode, "system"),
        }),
    );
    session.text(frame.to_string()).await.is_ok()
}

fn episode_status(episode: &Value) -> &str {
    episode.get("status").and_then(Value::as_str).unwrap_or_default()
}

fn merge(target: &mut Value, source: &Value) {
    let (Some(target), Some(source)) = (target.as_object_mut(), source.as_object()) else {
        return;
    };
    for (key, value) in source {
        target.insert(key.clone(), value.clone());
    }
}

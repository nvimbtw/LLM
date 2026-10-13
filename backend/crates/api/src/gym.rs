//! Drive browser-observed gym runs while the gym service owns episodes, steps and grades.

mod episode;
mod protocol;
mod steps;

use crate::{
    streaming::{envelope, string_at},
    ws,
};
use episode::Driver;
use protocol::Episode;
use serde_json::{json, Value};
pub use steps::play;
use uuid::Uuid;

/// Drive eligible episodes in order and report the run's final score to the socket.
/// A failed client delivery abandons the run; already graded episodes remain stored.
pub async fn run(
    session: &mut actix_ws::Session,
    engine: &llm_engine::Proxy,
    gyms: &gyms_engine::Proxy,
    caller: &auth::Caller,
    run_id: Uuid,
    ask: &Value,
) {
    let record = match gyms.get(caller, &format!("/gym/runs/{run_id}")).await {
        Ok(record) => record,
        Err(error) => {
            let _ = session.text(ws::error_frame(&error.to_string())).await;
            return;
        }
    };
    let model = string_at(&record, "model_name");
    let driver = Driver {
        engine,
        gyms,
        caller,
        model: &model,
        ask,
        observation_budget_chars: tools::Config::from_env().bash_budget_chars,
    };
    set_status(gyms, caller, run_id, "running").await;
    let episodes = record["episodes"].as_array().cloned().unwrap_or_default();
    for episode in episodes
        .iter()
        .map(Episode::from_value)
        .filter(Episode::runnable)
    {
        if session.text(episode.start_frame()).await.is_err() {
            set_status(gyms, caller, run_id, "abandoned").await;
            return;
        }
        if driver.episode(session, &episode).await.is_none() {
            close(gyms, caller, &episode.id).await;
            set_status(gyms, caller, run_id, "abandoned").await;
            return;
        }
    }
    complete(session, gyms, caller, run_id).await;
}

/// Mark a run done and send its score summary when the final read succeeds; preserve best-effort status writes.
async fn complete(
    session: &mut actix_ws::Session,
    gyms: &gyms_engine::Proxy,
    caller: &auth::Caller,
    run_id: Uuid,
) {
    set_status(gyms, caller, run_id, "done").await;
    if let Ok(finished) = gyms.get(caller, &format!("/gym/runs/{run_id}")).await {
        let frame = envelope(
            "run_done",
            &json!({"score":finished["score"],"correct":finished["correct"],
            "graded":finished["graded"],"episode_count":finished["episode_count"]}),
        );
        let _ = session.text(frame.to_string()).await;
    }
}

/// Best-effort close of an abandoned episode, retaining any step and grade records already stored.
async fn close(gyms: &gyms_engine::Proxy, caller: &auth::Caller, episode_id: &str) {
    let _ = gyms
        .post(
            caller,
            &format!("/gym/episodes/{episode_id}/close"),
            Value::Null,
        )
        .await;
}

/// Attempt the established run-status transition without changing the driver's legacy failure policy.
async fn set_status(gyms: &gyms_engine::Proxy, caller: &auth::Caller, run_id: Uuid, status: &str) {
    let _ = gyms
        .post(
            caller,
            &format!("/gym/runs/{run_id}/status"),
            json!({"status":status}),
        )
        .await;
}

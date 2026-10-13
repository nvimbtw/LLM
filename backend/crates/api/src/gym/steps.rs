//! Episode commands and the shared serving observation renderer.

use crate::streaming::string_at;
use serde_json::{json, Value};

/// Execute a shell call in its episode workspace and attach the serving renderer's observation.
/// Unsupported tools return a caveat without a step; service failures propagate to the caller.
pub async fn play(
    gyms: &gyms_engine::Proxy,
    caller: &auth::Caller,
    episode_id: &str,
    tool: &str,
    argument: &str,
    reply: &str,
    budget: usize,
) -> Result<Value, gyms_engine::GymsEngineError> {
    if tool != tools::Bash::NAME {
        let caveat = format!("There is no {tool} tool here. This gym has a shell.");
        return Ok(json!({"observation":tools::markup::warning(&caveat)}));
    }
    let mut ran = gyms
        .post(
            caller,
            &format!("/gym/episodes/{episode_id}/step"),
            json!({"command":argument,"reply":reply}),
        )
        .await?;
    let observation = render_observation(&ran, budget);
    if let Some(object) = ran.as_object_mut() {
        object.insert("observation".into(), json!(observation));
    }
    Ok(ran)
}

/// Render the shell fields under the existing permissive service contract and character budget.
fn render_observation(ran: &Value, budget: usize) -> String {
    tools::observation(
        tools::Output {
            stdout: string_at(ran, "stdout"),
            stderr: string_at(ran, "stderr"),
            exit_code: ran["exit_code"].as_i64().unwrap_or_default() as i32,
            timed_out: ran["timed_out"].as_bool().unwrap_or_default(),
        },
        budget,
    )
}

/// Run one command and return the rendered observation, converting transport failures into model-visible caveats.
pub(super) async fn execute(
    gyms: &gyms_engine::Proxy,
    caller: &auth::Caller,
    episode_id: &str,
    tool: &str,
    command: &str,
    reply: &str,
    budget: usize,
) -> String {
    match play(gyms, caller, episode_id, tool, command, reply, budget).await {
        Ok(ran) => string_at(&ran, "observation"),
        Err(error) => tools::markup::warning(&format!("The shell could not be reached: {error}")),
    }
}

//! Turning a gym's expert trajectories into a dataset that can be trained on.
//!
//! Three services and one job. The gyms service plays the trajectory and knows
//! what the shell said; this crate knows how that is *rendered* for a model,
//! because `tools::observation` is the same function a live round is served by;
//! the data engine owns every corpus anything trains on. The two Python
//! services never speak, so the assembly happens here — which is not a
//! compromise but the only place all three facts are available at once.
//!
//! **What is rendered here and what is not.** The observation body is rendered
//! here, in Rust, because a body rendered anywhere else is a second copy of a
//! layout the model is served by (`contracts/shell-observation.json` is what
//! that costs). No `<|tool-call|>` marker is spelled here at all: a turn that
//! makes a call carries `tool` and `argument` beside its text, and
//! `llm.chat.tools` renders the marker when the corpus is built. Spelling it
//! here would be a second copy of `llm.chat.tools.call` — and one that would
//! not even agree, because `json.dumps` writes `{"tool": "bash", …}` and
//! `serde_json` writes `{"tool":"bash",…}`, so the model would be trained on
//! bytes it is never served.
//!
//! So nothing is stored pre-rendered that this side could not verify, and the
//! demonstration reads back through the ordinary SFT path with no new reader in
//! the engine at all.
//!
//! **The assistant turn names the call, and that is the whole point of this
//! module.** SFT trains every assistant turn and masks every tool turn
//! (`llm.chat.template.training_segments`), so where the call is stored decides
//! whether the model is ever taught to make one. This used to hang the call on
//! the tool turn alone, which put it wholly inside the masked region: the
//! trained sequences were `"Searching every file for the marker."` followed by
//! `<|/assistant|>`, teaching a model to narrate a search and then close its
//! turn without reaching for the shell — the exact habit the gym exists to
//! train out, and the one behaviour these demonstrations are for.
//!
//! A live chat stores the same shape, which is the check this is held to: the
//! engine keeps an assistant reply as the raw text the model produced, markers
//! and all (`api/routes/chats.py`), so a call sits inside the assistant turn
//! there too. The call then appears twice in a rendered conversation — once as
//! something the model produced, once as the header of the result answering it
//! — and that repetition is deliberate; see `render_result`.

use serde_json::{Value, json};

use crate::turn;

/// The gyms service's `bash` tool is this project's `bash` tool. Named once
/// rather than threaded through: a demonstration that claimed some other tool
/// would be dropped by the engine's build, which refuses a call it has no tool
/// for.
const TOOL: &str = "bash";

/// One trajectory as a conversation, or `None` if it is not trainable.
///
/// A demonstration must end on the answer. The engine drops a conversation that
/// does not — a trailing tool result is a turn nothing is trained on, so the
/// conversation would cost context and teach no ending at all — so a trajectory
/// with no answer is dropped here, where the count it changes is still visible
/// in this request's response.
pub fn conversation(trajectory: &Value, budget: usize) -> Option<Value> {
    let answer = turn::string_at(trajectory, "answer");
    if answer.trim().is_empty() {
        return None;
    }

    let mut messages: Vec<Value> = Vec::new();

    let system = turn::string_at(trajectory, "system");
    if !system.trim().is_empty() {
        messages.push(json!({ "role": "system", "content": system }));
    }
    messages.push(json!({ "role": "user", "content": turn::string_at(trajectory, "prompt") }));

    for move_ in trajectory["moves"].as_array().into_iter().flatten() {
        let command = turn::string_at(move_, "command");
        // The reply is what the model would have said before calling — kept
        // even when it is empty, because this turn is trained either way. An
        // empty one teaches calling with no preamble, which is a fine thing to
        // teach; a *missing* one leaves two tool results adjacent, which is a
        // shape the runtime never emits.
        //
        // It names the call it is about to make, which is what puts the command
        // in the trained region rather than in the masked observation that
        // follows it. The text and the marker are not concatenated here — the
        // engine appends the rendered call to this content, for the reason the
        // module doc gives.
        messages.push(json!({
            "role": "assistant",
            "content": turn::string_at(move_, "reply"),
            "tool": TOOL,
            "argument": command,
        }));
        messages.push(json!({
            "role": "tool",
            "tool": TOOL,
            "argument": command,
            "content": tools::observation(
                tools::Output {
                    stdout: turn::string_at(move_, "stdout"),
                    stderr: turn::string_at(move_, "stderr"),
                    exit_code: move_["exit_code"].as_i64().unwrap_or_default() as i32,
                    timed_out: move_["timed_out"].as_bool().unwrap_or_default(),
                },
                budget,
            ),
        }));
    }

    messages.push(json!({ "role": "assistant", "content": answer }));

    Some(json!({
        "messages": messages,
        "meta": {
            "task_type": turn::string_at(trajectory, "task_type"),
            "seed": trajectory["seed"].clone(),
        },
    }))
}

/// What produced a set, recorded on the dataset the way a derived one records
/// its pipeline.
///
/// `max_output_chars` is the one that is easy to leave out and should not be. A
/// body is clamped when it is written, so a set rendered at one budget is not
/// interchangeable with one rendered at another — and nothing in the engine's
/// corpus key can see the difference, because the key is a function of the
/// dataset's id, count and timestamp.
pub fn provenance(gym_id: &str, task_types: &Value, seed_base: i64, budget: usize) -> Value {
    json!([{
        "gym": gym_id,
        "task_types": task_types.clone(),
        "seed_base": seed_base,
        "max_output_chars": budget,
        "source": "expert",
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trajectory() -> Value {
        json!({
            "task_type": "bash_grep",
            "seed": 4,
            "system": "You have a shell.",
            "prompt": "Which file contains ANSWER_2ee433?",
            "moves": [
                {
                    "reply": "<|par|>Looking around.<|/par|>",
                    "command": "ls -la",
                    "stdout": "notes.txt\nother.txt",
                    "stderr": "",
                    "exit_code": 0,
                    "timed_out": false
                },
                {
                    "reply": "<|par|>Searching.<|/par|>",
                    "command": "grep -rn ANSWER_2ee433 .",
                    "stdout": "./notes.txt:3:ANSWER_2ee433",
                    "stderr": "",
                    "exit_code": 0,
                    "timed_out": false
                }
            ],
            "answer": "notes.txt"
        })
    }

    fn messages(value: &Value) -> Vec<Value> {
        value["messages"].as_array().unwrap().clone()
    }

    #[test]
    fn a_trajectory_becomes_the_conversation_the_template_reads() {
        let built = conversation(&trajectory(), 4000).expect("trainable");
        let turns = messages(&built);
        let roles: Vec<&str> = turns.iter().map(|m| m["role"].as_str().unwrap()).collect();

        assert_eq!(
            roles,
            vec!["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
        );
    }

    #[test]
    fn a_demonstration_ends_on_the_answer_alone() {
        // The whole lesson. The first measured runs produced a model that put
        // the answer on screen and then narrated more searching; nothing it had
        // been trained on ended with a bare answer.
        let built = conversation(&trajectory(), 4000).expect("trainable");
        let last = messages(&built).last().unwrap().clone();

        assert_eq!(last["role"], "assistant");
        assert_eq!(last["content"], "notes.txt");
    }

    #[test]
    fn the_assistant_turn_that_calls_the_shell_names_the_call() {
        // The one that decides whether these demonstrations teach tool use at
        // all. SFT trains assistant turns and masks tool turns, so a command
        // stored only on the tool turn carries no gradient: the model would be
        // trained on the narration and never on the reaching-for-the-shell.
        //
        // The marker itself is deliberately absent — `llm.chat.tools.call`
        // spells it, and a copy here would disagree with it byte for byte.
        let built = conversation(&trajectory(), 4000).expect("trainable");
        let calling: Vec<Value> = messages(&built)
            .into_iter()
            .filter(|m| m["role"] == "assistant" && !m["tool"].is_null())
            .collect();

        assert_eq!(calling.len(), 2, "one per move");
        assert_eq!(calling[0]["tool"], "bash");
        assert_eq!(calling[0]["argument"], "ls -la");
        assert_eq!(calling[1]["argument"], "grep -rn ANSWER_2ee433 .");
        for turn in &calling {
            assert!(!turn["content"].as_str().unwrap().contains("<|tool-call|>"));
        }
    }

    #[test]
    fn the_answer_turn_names_no_call_because_it_makes_none() {
        // The last turn ends the episode. A call on it would render a marker
        // after the answer, and the grader reads the last line.
        let built = conversation(&trajectory(), 4000).expect("trainable");
        let last = messages(&built).last().unwrap().clone();

        assert!(last["tool"].is_null());
        assert!(last["argument"].is_null());
    }

    #[test]
    fn a_call_is_stored_identically_on_both_turns_it_appears_on() {
        // A rendered conversation shows the call twice — once as something the
        // model produced, once as the header of the result answering it. They
        // are the same call, and a demonstration whose two copies disagreed
        // would teach the model that its own command came back altered.
        let built = conversation(&trajectory(), 4000).expect("trainable");
        let turns = messages(&built);
        let pairs: Vec<(&Value, &Value)> = turns
            .windows(2)
            .filter(|w| w[0]["role"] == "assistant" && w[1]["role"] == "tool")
            .map(|w| (&w[0], &w[1]))
            .collect();

        assert_eq!(pairs.len(), 2);
        for (assistant, tool) in pairs {
            assert_eq!(assistant["tool"], tool["tool"]);
            assert_eq!(assistant["argument"], tool["argument"]);
        }
    }

    #[test]
    fn a_tool_turn_carries_its_call_and_an_unwrapped_body() {
        // Stored the way `ChatTurn` names a call, so the engine's single
        // renderer adds the header at build time. A `<|tool-call|>` spelled here
        // would be a fourth copy of `tools::call`.
        let built = conversation(&trajectory(), 4000).expect("trainable");
        let tool = messages(&built)
            .into_iter()
            .find(|m| m["role"] == "tool")
            .expect("a tool turn");

        assert_eq!(tool["tool"], "bash");
        assert_eq!(tool["argument"], "ls -la");
        assert!(!tool["content"].as_str().unwrap().contains("<|tool-call|>"));
    }

    #[test]
    fn the_body_is_what_a_live_round_would_have_shown() {
        // The reason this assembly is in Rust at all, stated as a test.
        let built = conversation(&trajectory(), 4000).expect("trainable");
        let tool = messages(&built)
            .into_iter()
            .find(|m| m["role"] == "tool")
            .expect("a tool turn");

        let live = tools::observation(
            tools::Output {
                stdout: "notes.txt\nother.txt".to_string(),
                stderr: String::new(),
                exit_code: 0,
                timed_out: false,
            },
            4000,
        );
        assert_eq!(tool["content"].as_str().unwrap(), live);
    }

    #[test]
    fn a_trajectory_with_no_answer_is_dropped_rather_than_stored() {
        let mut without = trajectory();
        without["answer"] = json!("   ");

        assert!(conversation(&without, 4000).is_none());
    }

    #[test]
    fn every_command_is_answered_so_no_two_results_land_adjacent() {
        // A missing assistant turn between two calls is a shape the runtime
        // never emits, and the template would render it as one.
        let built = conversation(&trajectory(), 4000).expect("trainable");
        let turns = messages(&built);
        let roles: Vec<&str> = turns.iter().map(|m| m["role"].as_str().unwrap()).collect();

        assert!(roles.windows(2).all(|pair| pair != ["tool", "tool"]));
    }

    #[test]
    fn the_provenance_records_the_budget_the_bodies_were_clamped_at() {
        let steps = provenance("bash", &json!(["bash_grep"]), 7, 4000);

        assert_eq!(steps[0]["max_output_chars"], 4000);
        assert_eq!(steps[0]["seed_base"], 7);
        assert_eq!(steps[0]["gym"], "bash");
    }
}

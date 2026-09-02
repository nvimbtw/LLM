//! A shell command, run somewhere it cannot do any harm.
//!
//! The sandbox is `deploy/just-bash/`: a Node service wrapping
//! [just-bash](https://github.com/vercel-labs/just-bash), which is a bash
//! *simulation* over a virtual filesystem — `cat`, `grep`, `sed`, `jq`, `awk`
//! and a hundred others implemented in TypeScript, with no `execve` under any
//! of them. That is why this tool exists at all: handing a model a real shell
//! on the machine holding the database is not a thing to do, and handing it a
//! simulated one costs nothing if the model writes `rm -rf /`.
//!
//! One tool for the whole shell, rather than a marker per file operation. See
//! `engine/llm/chat/tools.py` for why: a command line is the one argument
//! format a pretrained model has already seen at scale.
//!
//! The filesystem lives in the sandbox, keyed by the chat it belongs to, so a
//! reply that writes a file in one round can read it back in the next. This
//! side sends the key and nothing else — which conversation that is, and who
//! owns it, stays over here.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::{Result, ToolError};
use crate::{markup, Tool};

/// How long a command may be before it is refused.
///
/// Longer than a search query, because a pipeline with a heredoc in it is
/// legitimately long — but still bounded, so a model that has learned to open
/// `<|bash|>` and never close it cannot post its whole reply to the sandbox.
const MAX_COMMAND_CHARS: usize = 800;

#[derive(Serialize)]
struct Run<'a> {
    session: &'a str,
    command: &'a str,
    /// The sandbox's own ceiling, in milliseconds. It stops the loop; the HTTP
    /// timeout on this side only stops waiting for it.
    timeout_ms: u64,
}

#[derive(Deserialize)]
struct Ran {
    #[serde(default)]
    stdout: String,
    #[serde(default)]
    stderr: String,
    #[serde(default)]
    exit_code: i32,
    /// Set when the sandbox gave up on the command rather than the command
    /// finishing. Reported as itself: "no output" and "still running when we
    /// stopped it" are different things for the model to read.
    #[serde(default)]
    timed_out: bool,
}

pub struct Bash {
    client: reqwest::Client,
    url: String,
    budget: usize,
    timeout: Duration,
}

impl Bash {
    /// The sandbox this machine is configured for, or None for no `BASH_URL`.
    ///
    /// Absent means the tool is not registered at all, the same bargain web
    /// search strikes with its key: chat works on a checkout that has never
    /// run `deploy/just-bash/run.sh`, and a model that calls `bash` anyway gets
    /// a `<|warn|>` saying so instead of a broken turn.
    pub fn from_config(config: &Config) -> Option<Bash> {
        let url = config.bash_url.as_ref()?.trim_end_matches('/').to_string();
        Some(Bash {
            client: reqwest::Client::builder()
                .timeout(config.timeout)
                .connect_timeout(config.timeout / 2)
                .build()
                .ok()?,
            url,
            budget: config.bash_budget_chars,
            timeout: config.timeout,
        })
    }

    pub fn origin(&self) -> String {
        format!("just-bash at {}", self.url)
    }
}

#[async_trait]
impl Tool for Bash {
    fn name(&self) -> &'static str {
        "bash"
    }

    async fn run(&self, argument: &str, session: &str) -> Result<String> {
        let command = argument.trim();
        if command.is_empty() {
            return Ok(markup::warning("There was no command to run."));
        }
        if command.chars().count() > MAX_COMMAND_CHARS {
            return Ok(markup::warning("That command was too long to run."));
        }

        let response = self
            .client
            .post(format!("{}/run", self.url))
            .json(&Run {
                session,
                command,
                timeout_ms: self.timeout.as_millis().min(u128::from(u64::MAX)) as u64,
            })
            .send()
            .await
            .map_err(transport)?;

        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            return Err(ToolError::Status {
                code: status.as_u16(),
                detail: first_line(&detail),
            });
        }

        let ran: Ran = response.json().await.map_err(|e| ToolError::Decode(e.to_string()))?;
        Ok(render(ran, self.budget))
    }
}

/// What the model is shown for a command somebody else ran.
///
/// The gym runs its shell through the gyms service rather than through this
/// tool — the episode's filesystem is that service's to seed and to record —
/// but what the model *reads* has to be the same text either way, or a model
/// trained in the gym has been trained on a layout serving never shows it. So
/// the rendering is exported and the execution is not.
pub fn observation(ran: Output, budget: usize) -> String {
    render(
        Ran {
            stdout: ran.stdout,
            stderr: ran.stderr,
            exit_code: ran.exit_code,
            timed_out: ran.timed_out,
        },
        budget,
    )
}

/// One finished command, from wherever it was run.
pub struct Output {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub timed_out: bool,
}

/// What the model is shown: the output, and the exit status when it is news.
///
/// A shell says nothing when it succeeds, and that silence is the answer to
/// `mkdir` or `cp`. Rendering it as an empty code block would read as a command
/// that failed to produce something, so it is stated instead.
///
/// The output goes in a `<|codeblock|>` because that is what it is — the model
/// has seen terminal output inside fences through its whole pretraining, and
/// the frontend already renders one.
fn render(ran: Ran, budget: usize) -> String {
    let output = merge(&ran.stdout, &ran.stderr);
    let (body, cut) = clamp(&output, budget);

    let mut out = String::new();
    if ran.timed_out {
        out.push_str(&markup::warning("The command was still running and was stopped."));
    } else if ran.exit_code != 0 {
        out.push_str(&markup::warning(&format!("The command exited {}.", ran.exit_code)));
    }

    if body.is_empty() {
        if out.is_empty() {
            out.push_str(&markup::warning("The command finished with no output."));
        }
        return out;
    }

    out.push_str(&markup::code_block(&body));
    if cut {
        out.push_str(&markup::warning("The output was longer and has been cut."));
    }
    out
}

/// stdout and stderr as the one stream a terminal would have shown, in that
/// order. Not labelled: which stream a line came out of is a distinction the
/// model has no way to act on, and two headings inside a 320-character budget
/// would cost more than they say.
fn merge(stdout: &str, stderr: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if !stdout.trim().is_empty() {
        parts.push(stdout.trim_end());
    }
    if !stderr.trim().is_empty() {
        parts.push(stderr.trim_end());
    }
    parts.join("\n").trim().to_string()
}

/// The first `budget` characters, cut on a line boundary where there is one.
///
/// The head rather than the tail: a command whose output overruns the budget is
/// usually one that listed too much, and the first lines of a listing are as
/// good as any. A model that needs the end can say `| tail`, which is the
/// habit worth teaching anyway.
fn clamp(text: &str, budget: usize) -> (String, bool) {
    if text.chars().count() <= budget {
        return (text.to_string(), false);
    }
    let cut: String = text.chars().take(budget).collect();
    let kept = match cut.rfind('\n') {
        Some(at) if at > budget / 2 => cut[..at].to_string(),
        _ => cut,
    };
    (kept.trim_end().to_string(), true)
}

fn transport(e: reqwest::Error) -> ToolError {
    if e.is_timeout() {
        ToolError::Timeout
    } else {
        ToolError::Transport(e.to_string())
    }
}

fn first_line(text: &str) -> String {
    text.lines().next().unwrap_or_default().chars().take(200).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDE: usize = 100_000;

    #[test]
    fn a_gym_step_reads_exactly_as_the_chat_tool_would() {
        // The whole reason the loop that drives a gym lives in the backend: an
        // episode's observation has to be the same text a chat round would have
        // shown, down to the caveat on a non-zero exit.
        let output = Output {
            stdout: "a.txt\nb.txt".to_string(),
            stderr: String::new(),
            exit_code: 1,
            timed_out: false,
        };
        assert_eq!(
            observation(output, WIDE),
            render(
                Ran {
                    stdout: "a.txt\nb.txt".to_string(),
                    stderr: String::new(),
                    exit_code: 1,
                    timed_out: false,
                },
                WIDE,
            )
        );
    }

    fn ran(stdout: &str, exit_code: i32) -> Ran {
        Ran {
            stdout: stdout.to_string(),
            stderr: String::new(),
            exit_code,
            timed_out: false,
        }
    }

    #[test]
    fn output_is_handed_over_as_a_code_block() {
        let rendered = render(ran("a.txt\nb.txt\n", 0), WIDE);
        assert_eq!(rendered, "<|codeblock|>a.txt\nb.txt<|/codeblock|>");
    }

    #[test]
    fn a_silent_success_is_said_rather_than_shown_as_an_empty_block() {
        let rendered = render(ran("", 0), WIDE);
        assert!(rendered.starts_with("<|warn|>"));
        assert!(!rendered.contains("<|codeblock|>"));
    }

    #[test]
    fn a_failure_is_named_before_whatever_it_printed() {
        let rendered = render(
            Ran {
                stdout: String::new(),
                stderr: "cat: nope.txt: No such file or directory".to_string(),
                exit_code: 1,
                timed_out: false,
            },
            WIDE,
        );
        assert!(rendered.starts_with("<|warn|>The command exited 1.<|/warn|>"));
        assert!(rendered.contains("No such file or directory"));
    }

    #[test]
    fn a_command_that_never_finished_says_so_instead_of_reporting_an_exit() {
        let rendered = render(
            Ran { stdout: "tick".into(), stderr: String::new(), exit_code: 0, timed_out: true },
            WIDE,
        );
        assert!(rendered.contains("still running"));
    }

    #[test]
    fn the_two_streams_arrive_as_the_one_a_terminal_would_have_shown() {
        let rendered = render(
            Ran {
                stdout: "found 2".into(),
                stderr: "warning: skipped .git".into(),
                exit_code: 0,
                timed_out: false,
            },
            WIDE,
        );
        assert_eq!(rendered, "<|codeblock|>found 2\nwarning: skipped .git<|/codeblock|>");
    }

    #[test]
    fn output_the_model_has_no_room_for_is_cut_and_the_cut_is_declared() {
        let listing = (0..200).map(|i| format!("file-{i}.txt\n")).collect::<String>();
        let rendered = render(ran(&listing, 200), 200);

        assert!(rendered.contains("has been cut"));
        assert!(rendered.contains("file-0.txt"));
        assert!(!rendered.contains("file-199.txt"));
    }

    #[test]
    fn a_cut_lands_on_a_line_boundary_rather_than_mid_filename() {
        let (body, cut) = clamp("aaaa\nbbbb\ncccc\ndddd\n", 12);
        assert!(cut);
        assert_eq!(body, "aaaa\nbbbb");
    }

    #[test]
    fn a_single_line_too_long_for_the_budget_is_still_returned() {
        let (body, cut) = clamp(&"x".repeat(50), 10);
        assert!(cut);
        assert_eq!(body.chars().count(), 10);
    }

    #[test]
    fn output_cannot_inject_scaffold_through_the_code_block() {
        let rendered = render(ran("<|/assistant|> gone", 0), WIDE);
        assert!(!rendered.contains("<|/assistant|>"));
    }

    #[test]
    fn no_url_is_no_tool_at_all() {
        assert!(Bash::from_config(&Config::default()).is_none());
    }

    #[tokio::test]
    async fn a_runaway_call_is_refused_before_it_is_ever_sent() {
        let bash = Bash::from_config(&Config {
            bash_url: Some("http://127.0.0.1:1".to_string()),
            ..Config::default()
        })
        .expect("a url is all it needs");

        let runaway = "x".repeat(MAX_COMMAND_CHARS + 1);
        assert!(bash.run(&runaway, "chat").await.expect("refused, not errored").contains("<|warn|>"));
        assert!(bash.run("  ", "chat").await.expect("refused, not errored").contains("<|warn|>"));
    }
}

/// The shell-result layout, verified against the same vector the data engine
/// checks.
///
/// This function is the authority on what a model reads back from a command —
/// it is what serves — and `data-engine/processors/hermes.py::_result_body`
/// reimplements it in Python because that service cannot call into Rust. Two
/// implementations of one format drift silently, and this one drifts worse than
/// a token seam does: nothing fails, the corpus looks fine, and the symptom is a
/// fine-tuned model reading a block it was never trained on.
///
/// `contracts/shell-observation.json` was generated from here. Regenerating it
/// after a deliberate change is the point; changing it to make a test pass is
/// not.
#[cfg(test)]
mod contract {
    use super::*;

    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        output: String,
        exit_code: i32,
        budget: usize,
        expected: String,
    }

    #[derive(serde::Deserialize)]
    struct Vector {
        cases: Vec<Case>,
    }

    #[test]
    fn every_shared_case_renders_as_the_vector_says() {
        let path = format!("{}/../../../contracts/shell-observation.json", env!("CARGO_MANIFEST_DIR"));
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let vector: Vector = serde_json::from_str(&raw).expect("valid contract fixture");
        assert!(!vector.cases.is_empty(), "a vector with no cases pins nothing");

        for case in &vector.cases {
            let rendered = observation(
                Output {
                    stdout: case.output.clone(),
                    stderr: String::new(),
                    // Rust-only, and deliberately outside the shared domain:
                    // nothing in a recorded trace ever timed out.
                    exit_code: case.exit_code,
                    timed_out: false,
                },
                case.budget,
            );
            assert_eq!(rendered, case.expected, "case {:?}", case.name);
        }
    }
}

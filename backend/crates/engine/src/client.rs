//! Typed calls into the Python engine.
//!
//! Everything above this crate calls these functions and never learns there is a
//! subprocess, an HTTP hop, or a JSON body involved. The request types mirror the
//! engine's pydantic models; the loose `serde_json::Value` payloads are the ones
//! whose shape is genuinely open (diagnostics, run results), where a Rust struct
//! would have to be edited every time a metric is added.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_tungstenite::tungstenite::Message;

use crate::chat::{SamplingParams, ThinkBudget, Turn};

#[derive(Debug)]
pub enum EngineError {
    /// The engine could not be reached at all.
    Transport(String),
    /// The engine answered, and refused.
    Status { code: u16, detail: String },
    /// The engine answered with something this build cannot read.
    Decode(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::Transport(e) => write!(f, "engine unreachable: {e}"),
            EngineError::Status { code, detail } => write!(f, "engine returned {code}: {detail}"),
            EngineError::Decode(e) => write!(f, "unreadable engine response: {e}"),
        }
    }
}

impl std::error::Error for EngineError {}

pub type Result<T> = std::result::Result<T, EngineError>;

// ── Requests ────────────────────────────────────────────────────────────────
//
// Defaults are the engine's, not repeated here: `skip_serializing_if` omits an
// unset field entirely so the engine fills it in. Restating them would give two
// places to change a default and one of them would be missed.

#[derive(Debug, Default, Serialize)]
pub struct TrainRequest {
    pub model_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heads: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ff_multiplier: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ctx_window: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_vocab: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epochs: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup_steps: Option<i64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub datasets: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub curriculum: Vec<CurriculumStage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shuffle: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rebuild_dataset: Option<bool>,
}

#[derive(Debug, Default, Serialize)]
pub struct SftRequest {
    pub model_name: String,
    pub base_model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epochs: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup_steps: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strip_thinking: Option<bool>,
    // Which SFT sets under the engine's sft/ to fine-tune on. Empty is not "none"
    // — the engine reads it as "every set on disk", so a set appearing there
    // later silently joins the next run. Send an explicit selection.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub datasets: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rebuild_dataset: Option<bool>,
}

#[derive(Debug, Default, Serialize)]
pub struct EvalRequest {
    pub model_name: String,
    /// "language" | "markup" | "assistant". Separate suites because a base model
    /// has never seen the chat scaffold, and folding those stages into one number
    /// would mark every pretrain down for failing a task it was not trained on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suite: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

/// One assistant turn, generated from a whole conversation through the engine's
/// canonical template and under the marker grammar.
#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model_name: String,
    pub turns: Vec<Turn>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget: Option<ThinkBudget>,
    pub max_tokens: i64,
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
}

impl ChatRequest {
    pub fn new(model_name: impl Into<String>, turns: Vec<Turn>, max_tokens: i64) -> ChatRequest {
        let sampling = SamplingParams::default();
        ChatRequest {
            model_name: model_name.into(),
            turns,
            system: None,
            budget: None,
            max_tokens,
            temperature: sampling.temperature,
            top_k: sampling.top_k,
            top_p: sampling.top_p,
        }
    }

    pub fn with_sampling(mut self, sampling: SamplingParams) -> ChatRequest {
        self.temperature = sampling.temperature;
        self.top_k = sampling.top_k;
        self.top_p = sampling.top_p;
        self
    }
}

#[derive(Debug, Serialize)]
pub struct MeasureRequest {
    pub model_name: String,
    pub turns: Vec<Turn>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
}

#[derive(Debug, Serialize)]
struct CountRequest<'a> {
    model_name: &'a str,
    text: &'a str,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    /// Blocks the engine had to close because the reply hit its token cap.
    pub drained_blocks: usize,
    /// "marker" when the model ended its own turn, "length" when it ran out.
    pub stopped: String,
}

#[derive(Debug, Deserialize)]
pub struct TokenCount {
    pub tokens: usize,
}

#[derive(Debug, Deserialize)]
pub struct Measurement {
    pub tokens: usize,
    pub trainable_tokens: usize,
}

/// One corpus on the engine's disk, as the composer offers it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub name: String,
    pub files: usize,
    pub bytes: u64,
}

/// A trained model as the engine has it on disk.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub has_weights: bool,
    #[serde(default)]
    pub config: Value,
}

#[derive(Debug, Deserialize)]
struct ModelList {
    models: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct DatasetList {
    datasets: Vec<DatasetInfo>,
}

/// One curriculum stage: datasets trained for their own epoch count before the
/// run moves on. Order is the curriculum.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CurriculumStage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub datasets: Vec<String>,
    pub epochs: i64,
}

#[derive(Debug, Deserialize)]
pub struct Markers {
    pub markers: Vec<String>,
    pub block_size: usize,
    pub end_of_text: String,
}

#[derive(Debug, Default, Serialize)]
pub struct InferRequest {
    pub model_name: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

// ── Responses ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize)]
pub struct RunSnapshot {
    pub run_id: String,
    pub kind: String,
    pub status: String,
    pub elapsed_secs: f64,
    #[serde(default)]
    pub result: Option<Value>,
    #[serde(default)]
    pub error: Option<String>,
}

impl RunSnapshot {
    /// Terminal states, matching the engine's own set.
    pub fn is_terminal(&self) -> bool {
        matches!(self.status.as_str(), "completed" | "failed" | "cancelled")
    }
    pub fn is_cancelled(&self) -> bool {
        self.status == "cancelled"
    }
}

#[derive(Debug, Deserialize)]
pub struct InferResponse {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
}

/// One frame off a run's event stream.
#[derive(Debug, Clone)]
pub enum RunEvent {
    Progress(Value),
    Done(Value),
    Cancelled(Value),
    Error(String),
}

// ── Client ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    base_url: String,
}

impl Client {
    pub fn new(base_url: impl Into<String>) -> Client {
        Client {
            // No global timeout: a training run's event stream is open for as
            // long as the run lasts, and a request timeout would sever it
            // mid-run. Connect timeouts still apply.
            http: reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            base_url: base_url.into(),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// The same host and path, as a socket. Only the scheme differs, so the
    /// base URL stays the one knob configuring where the engine is.
    fn ws_url(&self, path: &str) -> String {
        let base = match self.base_url.split_once("://") {
            Some(("http", rest)) => format!("ws://{rest}"),
            Some(("https", rest)) => format!("wss://{rest}"),
            _ => self.base_url.clone(),
        };
        format!("{base}{path}")
    }

    pub async fn health(&self) -> Result<Value> {
        self.get("/").await
    }

    pub async fn train(&self, req: &TrainRequest) -> Result<RunSnapshot> {
        self.post("/train", req).await
    }

    /// Start a training run and stream it to completion over one socket.
    ///
    /// The whole run lives on this connection: the config goes up, every frame
    /// comes back as the engine emits it, and setting `cancel` sends a cancel
    /// message down the same socket rather than opening a second HTTP call that
    /// has to find the run again by id. Progress payloads are identical to the
    /// ones `follow` yields, so a caller reads either the same way.
    ///
    /// The returned snapshot is read back from `/runs/{id}` once the stream
    /// ends: the terminal frame says what happened, but the run itself is the
    /// authority on its final status.
    pub async fn train_stream<F>(
        &self,
        req: &TrainRequest,
        cancel: Arc<AtomicBool>,
        mut on_event: F,
    ) -> Result<RunSnapshot>
    where
        F: FnMut(RunEvent),
    {
        let (socket, _) = tokio_tungstenite::connect_async(self.ws_url("/train"))
            .await
            .map_err(|e| EngineError::Transport(e.to_string()))?;
        let (mut sink, mut stream) = socket.split();

        let opening = serde_json::json!({ "config": req }).to_string();
        sink.send(Message::text(opening))
            .await
            .map_err(|e| EngineError::Transport(e.to_string()))?;

        // Held until the socket closes: a run that never announced itself is a
        // run there is nothing to ask about afterwards.
        let mut run_id: Option<String> = None;
        let mut cancel_sent = false;
        let mut poll = tokio::time::interval(Duration::from_millis(250));

        loop {
            let frame = tokio::select! {
                _ = poll.tick(), if !cancel_sent => {
                    if cancel.load(Ordering::Relaxed) {
                        let _ = sink.send(Message::text(r#"{"type":"cancel"}"#)).await;
                        cancel_sent = true;
                    }
                    continue;
                }
                frame = stream.next() => frame,
            };

            let Some(frame) = frame else { break };
            let text = match frame.map_err(|e| EngineError::Transport(e.to_string()))? {
                Message::Text(text) => text,
                Message::Close(_) => break,
                // Ping/pong are answered under this API; nothing else is spoken.
                _ => continue,
            };

            let mut payload: Value = serde_json::from_str(text.as_str())
                .map_err(|e| EngineError::Decode(format!("{e}: {text}")))?;
            // Lifted off the payload rather than left on it, so what reaches the
            // caller is exactly the event body the SSE path delivers.
            let kind = payload
                .as_object_mut()
                .and_then(|frame| frame.remove("type"))
                .and_then(|kind| kind.as_str().map(str::to_string))
                .unwrap_or_default();

            match kind.as_str() {
                "started" => {
                    run_id = payload.get("run_id").and_then(|id| id.as_str()).map(str::to_string);
                }
                "progress" => on_event(RunEvent::Progress(payload)),
                "done" => {
                    on_event(RunEvent::Done(payload));
                    break;
                }
                "cancelled" => {
                    on_event(RunEvent::Cancelled(payload));
                    break;
                }
                "error" => {
                    let detail = payload
                        .get("error")
                        .and_then(|e| e.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    // Refused before a run existed — a busy engine or a config it
                    // would not take. That is this call failing, not a run
                    // failing, and it surfaces the way `train` used to.
                    if run_id.is_none() {
                        return Err(EngineError::Status { code: 409, detail });
                    }
                    on_event(RunEvent::Error(detail));
                    break;
                }
                _ => {}
            }
        }

        let _ = sink.close().await;
        match run_id {
            Some(run_id) => self.run(&run_id).await,
            None => Err(EngineError::Transport(
                "train socket closed before the run started".into(),
            )),
        }
    }

    pub async fn sft(&self, req: &SftRequest) -> Result<RunSnapshot> {
        self.post("/sft", req).await
    }

    pub async fn evaluate(&self, req: &EvalRequest) -> Result<RunSnapshot> {
        self.post("/eval", req).await
    }

    pub async fn infer(&self, req: &InferRequest) -> Result<InferResponse> {
        self.post("/infer", req).await
    }

    /// Generate the next assistant turn for a conversation.
    pub async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        self.post("/chat", req).await
    }

    /// Stream that turn, calling `on_token` with each decoded piece.
    ///
    /// The pieces are decoded in context by the engine, so concatenating them
    /// yields exactly the `text` of the final summary — a byte-level token can
    /// be half a UTF-8 character, and decoding one alone would not.
    pub async fn chat_stream<F>(&self, req: &ChatRequest, mut on_token: F) -> Result<ChatResponse>
    where
        F: FnMut(&str),
    {
        let response = self
            .http
            .post(self.url("/chat/stream"))
            .json(req)
            .send()
            .await
            .map_err(|e| EngineError::Transport(e.to_string()))?;

        if !response.status().is_success() {
            return Err(status_error(response).await);
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| EngineError::Transport(e.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(split) = buffer.find("\n\n") {
                let frame = buffer[..split].to_string();
                buffer.drain(..split + 2);

                let (kind, data) = split_frame(&frame);
                match kind.as_str() {
                    "token" => {
                        if let Some(piece) = data.get("token").and_then(|t| t.as_str()) {
                            on_token(piece);
                        }
                    }
                    "done" => {
                        return serde_json::from_value(data)
                            .map_err(|e| EngineError::Decode(e.to_string()))
                    }
                    "error" => {
                        return Err(EngineError::Status {
                            code: 500,
                            detail: data
                                .get("error")
                                .and_then(|e| e.as_str())
                                .unwrap_or("unknown")
                                .to_string(),
                        })
                    }
                    _ => {}
                }
            }
        }

        Err(EngineError::Transport("chat stream ended without a done frame".into()))
    }

    /// Token length of `text` under a model's tokenizer. Loads no weights, so
    /// this is cheap enough to call per message for usage accounting.
    pub async fn count_tokens(&self, model_name: &str, text: &str) -> Result<usize> {
        let count: TokenCount = self
            .post("/tokenize/count", &CountRequest { model_name, text })
            .await?;
        Ok(count.tokens)
    }

    /// Token cost of a conversation as SFT would render it.
    pub async fn measure(&self, req: &MeasureRequest) -> Result<Measurement> {
        self.post("/chat/measure", req).await
    }

    /// What corpora the engine has on disk.
    ///
    /// The engine owns the data; this side only names it in a curriculum. There
    /// is deliberately no upload counterpart — corpora arrive through the fetch
    /// scripts, not through the API.
    pub async fn datasets(&self) -> Result<Vec<DatasetInfo>> {
        let list: DatasetList = self.get("/datasets").await?;
        Ok(list.datasets)
    }

    /// What SFT sets the engine has on disk, for a caller composing a fine-tune.
    ///
    /// A separate catalogue from `datasets` rather than a flag on it: these are
    /// conversation sets under `sft/`, not prose corpora under the pretrain root,
    /// and a run picks from exactly one of the two.
    pub async fn sft_datasets(&self) -> Result<Vec<DatasetInfo>> {
        let list: DatasetList = self.get("/sft/datasets").await?;
        Ok(list.datasets)
    }

    /// Trained models on the engine's disk, with the architecture each carries.
    pub async fn models(&self) -> Result<Vec<ModelInfo>> {
        let list: ModelList = self.get("/models").await?;
        Ok(list.models)
    }

    /// The reserved marker vocabulary, so a caller derives its tag set from the
    /// real thing rather than keeping a second copy in sync.
    pub async fn markers(&self) -> Result<Markers> {
        self.get("/markers").await
    }

    pub async fn run(&self, run_id: &str) -> Result<RunSnapshot> {
        self.get(&format!("/runs/{run_id}")).await
    }

    /// Ask a run to stop at its next step boundary.
    ///
    /// A run that is already terminal answers 409, which is not an error here —
    /// cancelling something that has already stopped is the outcome the caller
    /// wanted.
    pub async fn cancel(&self, run_id: &str) -> Result<()> {
        let response = self
            .http
            .post(self.url(&format!("/runs/{run_id}/cancel")))
            .send()
            .await
            .map_err(|e| EngineError::Transport(e.to_string()))?;

        if response.status().is_success() || response.status().as_u16() == 409 {
            return Ok(());
        }
        Err(status_error(response).await)
    }

    /// Consume a run's event stream to its terminal frame, calling `on_event`
    /// for each one, then return the final snapshot.
    ///
    /// The stream replays whatever the run has already emitted, so a caller that
    /// attaches late — or reattaches after a restart — sees the whole run.
    pub async fn follow<F>(&self, run_id: &str, mut on_event: F) -> Result<RunSnapshot>
    where
        F: FnMut(RunEvent),
    {
        let response = self
            .http
            .get(self.url(&format!("/runs/{run_id}/events")))
            .send()
            .await
            .map_err(|e| EngineError::Transport(e.to_string()))?;

        if !response.status().is_success() {
            return Err(status_error(response).await);
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| EngineError::Transport(e.to_string()))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Frames are delimited by a blank line. A chunk can split one
            // anywhere, so only whole frames are taken and the remainder stays
            // buffered for the next chunk.
            while let Some(split) = buffer.find("\n\n") {
                let frame = buffer[..split].to_string();
                buffer.drain(..split + 2);
                if let Some(event) = parse_frame(&frame) {
                    let terminal = !matches!(event, RunEvent::Progress(_));
                    on_event(event);
                    if terminal {
                        // The stream is finished; the authoritative status comes
                        // from the run itself rather than being inferred from
                        // which frame arrived.
                        return self.run(run_id).await;
                    }
                }
            }
        }

        self.run(run_id).await
    }

    async fn get<T: for<'de> Deserialize<'de>>(&self, path: &str) -> Result<T> {
        let response = self
            .http
            .get(self.url(path))
            .send()
            .await
            .map_err(|e| EngineError::Transport(e.to_string()))?;
        decode(response).await
    }

    async fn post<B: Serialize, T: for<'de> Deserialize<'de>>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<T> {
        let response = self
            .http
            .post(self.url(path))
            .json(body)
            .send()
            .await
            .map_err(|e| EngineError::Transport(e.to_string()))?;
        decode(response).await
    }
}

async fn decode<T: for<'de> Deserialize<'de>>(response: reqwest::Response) -> Result<T> {
    if !response.status().is_success() {
        return Err(status_error(response).await);
    }
    let body = response
        .text()
        .await
        .map_err(|e| EngineError::Transport(e.to_string()))?;
    serde_json::from_str(&body).map_err(|e| EngineError::Decode(format!("{e}: {body}")))
}

/// Turns a failed response into an error carrying the engine's own message.
///
/// FastAPI puts it in `detail`; anything else is passed through verbatim rather
/// than replaced with a generic string, since the engine's message is the only
/// thing that says what actually went wrong.
async fn status_error(response: reqwest::Response) -> EngineError {
    let code = response.status().as_u16();
    let body = response.text().await.unwrap_or_default();
    let detail = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|v| v.get("detail").map(|d| d.to_string().trim_matches('"').to_string()))
        .unwrap_or(body);
    EngineError::Status { code, detail }
}

/// Splits an SSE frame into its event name and decoded payload.
fn split_frame(frame: &str) -> (String, Value) {
    let mut kind = String::new();
    let mut data = String::new();
    for line in frame.lines() {
        if let Some(rest) = line.strip_prefix("event: ") {
            kind = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("data: ") {
            data.push_str(rest);
        }
    }
    (kind, serde_json::from_str(&data).unwrap_or(Value::Null))
}

fn parse_frame(frame: &str) -> Option<RunEvent> {
    let mut kind = "";
    let mut data = String::new();
    for line in frame.lines() {
        if let Some(rest) = line.strip_prefix("event: ") {
            kind = rest.trim();
        } else if let Some(rest) = line.strip_prefix("data: ") {
            data.push_str(rest);
        }
    }
    let payload: Value = serde_json::from_str(&data).unwrap_or(Value::Null);
    match kind {
        "progress" => Some(RunEvent::Progress(payload)),
        "done" => Some(RunEvent::Done(payload)),
        "cancelled" => Some(RunEvent::Cancelled(payload)),
        "error" => Some(RunEvent::Error(
            payload.get("error").and_then(|e| e.as_str()).unwrap_or_default().to_string(),
        )),
        _ => None,
    }
}

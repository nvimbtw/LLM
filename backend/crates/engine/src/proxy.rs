//! Forwarding to a service.
//!
//! The backend does not model what a service owns. It authenticates a person,
//! signs a short-lived token saying who it acts for, and passes the request on
//! — so a field the service adds needs no change here, and a status it returns
//! reaches the client as itself rather than flattened into a generic failure.
//!
//! Nothing here names the engine. A second service is a second `Proxy` with its
//! own audience and its own routers above it; this is the whole seam.

use futures_util::{Stream, StreamExt};
use serde_json::Value;

use crate::{EngineError, Result};

pub use reqwest::Method;

/// One frame of an engine event stream.
#[derive(Clone, Debug)]
pub struct Frame {
    pub kind: String,
    pub data: Value,
}

impl Frame {
    /// Whether this frame ends the stream.
    pub fn is_final(&self) -> bool {
        matches!(self.kind.as_str(), "done" | "error" | "cancelled")
    }
}

#[derive(Clone)]
pub struct Proxy {
    http: reqwest::Client,
    base_url: String,
    /// Who the tokens are addressed to. A token minted for one service is
    /// refused by every other.
    audience: String,
    /// Unset means the engine is not enforcing authentication either — the
    /// development pair. A mismatch is a 401, which is the intended failure.
    secret: Option<String>,
}

impl Proxy {
    pub fn new(
        base_url: impl Into<String>,
        audience: impl Into<String>,
        secret: Option<String>,
    ) -> Proxy {
        Proxy {
            // No global timeout: a run's event stream is open for as long as
            // the run lasts. Connect timeouts still apply.
            http: reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            base_url: base_url.into(),
            audience: audience.into(),
            secret,
        }
    }

    pub async fn get(&self, caller: &auth::Caller, path: &str) -> Result<Value> {
        self.call(caller, Method::GET, path, None).await
    }

    pub async fn post(&self, caller: &auth::Caller, path: &str, body: Value) -> Result<Value> {
        self.call(caller, Method::POST, path, Some(body)).await
    }

    pub async fn patch(&self, caller: &auth::Caller, path: &str, body: Value) -> Result<Value> {
        self.call(caller, Method::PATCH, path, Some(body)).await
    }

    pub async fn put(&self, caller: &auth::Caller, path: &str, body: Value) -> Result<Value> {
        self.call(caller, Method::PUT, path, Some(body)).await
    }

    pub async fn delete(&self, caller: &auth::Caller, path: &str) -> Result<Value> {
        self.call(caller, Method::DELETE, path, None).await
    }

    pub async fn call(
        &self,
        caller: &auth::Caller,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value> {
        let mut request = self.http.request(method, self.url(path));
        if let Some(header) = self.authorization(caller) {
            request = request.header(reqwest::header::AUTHORIZATION, header);
        }
        if let Some(body) = body {
            request = request.json(&body);
        }

        let response =
            request.send().await.map_err(|e| EngineError::Transport(e.to_string()))?;
        if !response.status().is_success() {
            return Err(status_error(response).await);
        }

        let text = response.text().await.map_err(|e| EngineError::Transport(e.to_string()))?;
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text).map_err(|e| EngineError::Decode(format!("{e}: {text}")))
    }

    /// An engine event stream, frame by frame.
    ///
    /// The engine replays whatever a run has already emitted, so a caller that
    /// attaches late — or reattaches after a restart — sees the whole run.
    pub async fn events(
        &self,
        caller: &auth::Caller,
        path: &str,
    ) -> Result<impl Stream<Item = Result<Frame>>> {
        self.streaming(caller, Method::GET, path, None).await
    }

    /// The same, for a stream a request has to open with a body — a chat turn
    /// is asked and answered on one connection.
    pub async fn stream(
        &self,
        caller: &auth::Caller,
        path: &str,
        body: Value,
    ) -> Result<impl Stream<Item = Result<Frame>>> {
        self.streaming(caller, Method::POST, path, Some(body)).await
    }

    async fn streaming(
        &self,
        caller: &auth::Caller,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<impl Stream<Item = Result<Frame>>> {
        let mut request = self.http.request(method, self.url(path));
        if let Some(header) = self.authorization(caller) {
            request = request.header(reqwest::header::AUTHORIZATION, header);
        }
        if let Some(body) = body {
            request = request.json(&body);
        }

        let response =
            request.send().await.map_err(|e| EngineError::Transport(e.to_string()))?;
        if !response.status().is_success() {
            return Err(status_error(response).await);
        }

        Ok(frames(response.bytes_stream()))
    }

    /// A token for one call, or nothing when no secret is configured.
    ///
    /// Minted per request rather than cached: it lives for a minute, and a
    /// cached one would have to be refreshed on a schedule to save an HMAC.
    fn authorization(&self, caller: &auth::Caller) -> Option<String> {
        let secret = self.secret.as_ref()?;
        auth::token::issue_service_token(&self.audience, caller, &[], secret)
            .ok()
            .map(|token| format!("Bearer {token}"))
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }
}

/// Reassemble SSE frames from a byte stream.
///
/// Frames are delimited by a blank line, and a chunk can split one anywhere, so
/// only whole frames are emitted and the remainder stays buffered.
fn frames<B: AsRef<[u8]>>(
    bytes: impl Stream<Item = reqwest::Result<B>>,
) -> impl Stream<Item = Result<Frame>> {
    let mut buffer = String::new();
    let mut pending: Vec<Result<Frame>> = Vec::new();

    Box::pin(bytes).flat_map(move |chunk| {
        match chunk {
            Err(e) => pending.push(Err(EngineError::Transport(e.to_string()))),
            Ok(chunk) => {
                buffer.push_str(&String::from_utf8_lossy(chunk.as_ref()));
                while let Some(split) = buffer.find("\n\n") {
                    let frame = buffer[..split].to_string();
                    buffer.drain(..split + 2);
                    if let Some(frame) = parse(&frame) {
                        pending.push(Ok(frame));
                    }
                }
            }
        }
        futures_util::stream::iter(std::mem::take(&mut pending))
    })
}

fn parse(frame: &str) -> Option<Frame> {
    let mut kind = String::new();
    let mut data = String::new();
    for line in frame.lines() {
        if let Some(rest) = line.strip_prefix("event: ") {
            kind = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("data: ") {
            data.push_str(rest);
        }
    }
    if kind.is_empty() {
        return None;
    }
    Some(Frame { kind, data: serde_json::from_str(&data).unwrap_or(Value::Null) })
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

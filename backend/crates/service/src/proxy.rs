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

use crate::{Error, Result};

pub use reqwest::Method;

/// Signs a token when there is no shared secret: a carrier for the claims, not
/// a proof of anything.
const UNSIGNED: &str = "unsigned";

/// A file on its way to a service.
///
/// Named parts rather than a `Value`, because this is the one payload that is
/// not JSON and cannot be forwarded by the generic path — a service that takes
/// an upload takes it as multipart, and the field name is the service's to
/// choose.
pub struct FilePart {
    pub field: String,
    pub filename: String,
    pub media_type: String,
    pub bytes: Vec<u8>,
}

/// A file on its way back, with what the service said it is.
///
/// The headers travel with the bytes because only the service knows them: this
/// side stores nothing about the file and would otherwise have to guess a
/// content type from a name it never saw.
pub struct Blob {
    pub media_type: String,
    pub disposition: Option<String>,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Frame {
    pub kind: String,
    pub data: Value,
}

impl Frame {
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
    /// Unset means the service is not enforcing either — the development pair.
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

        self.read_json(request).await
    }

    async fn read_json(&self, request: reqwest::RequestBuilder) -> Result<Value> {
        let response = request
            .send()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;
        if !response.status().is_success() {
            return Err(status_error(response).await);
        }

        let text = response
            .text()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text).map_err(|e| Error::Decode(format!("{e}: {text}")))
    }

    /// A file, forwarded as multipart. The service answers with JSON like any
    /// other route — an upload is a write, and what it wrote is a row.
    pub async fn upload(&self, caller: &auth::Caller, path: &str, file: FilePart) -> Result<Value> {
        let part = reqwest::multipart::Part::bytes(file.bytes)
            .file_name(file.filename)
            .mime_str(&file.media_type)
            .map_err(|e| Error::Transport(e.to_string()))?;
        let form = reqwest::multipart::Form::new().part(file.field, part);

        let mut request = self.http.post(self.url(path)).multipart(form);
        if let Some(header) = self.authorization(caller) {
            request = request.header(reqwest::header::AUTHORIZATION, header);
        }
        self.read_json(request).await
    }

    /// A response that is bytes rather than JSON.
    ///
    /// Buffered, not streamed: what comes back this way is capped by the
    /// service that stores it, and a streaming body would have to carry the
    /// upstream's failure through a stream this side has already answered.
    pub async fn fetch(&self, caller: &auth::Caller, path: &str) -> Result<Blob> {
        let mut request = self.http.get(self.url(path));
        if let Some(header) = self.authorization(caller) {
            request = request.header(reqwest::header::AUTHORIZATION, header);
        }

        let response = request
            .send()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;
        if !response.status().is_success() {
            return Err(status_error(response).await);
        }

        let media_type = header_of(&response, reqwest::header::CONTENT_TYPE)
            .unwrap_or_else(|| "application/octet-stream".to_string());
        let disposition = header_of(&response, reqwest::header::CONTENT_DISPOSITION);
        let bytes = response
            .bytes()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;

        Ok(Blob {
            media_type,
            disposition,
            bytes: bytes.to_vec(),
        })
    }

    /// A service's event stream, frame by frame. It replays what a run has
    /// already emitted, so attaching late still sees the whole run.
    pub async fn events(
        &self,
        caller: &auth::Caller,
        path: &str,
    ) -> Result<impl Stream<Item = Result<Frame>>> {
        self.streaming(caller, Method::GET, path, None).await
    }

    /// The same for a stream that has to be opened with a body — a chat turn is
    /// asked and answered on one connection.
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

        let response = request
            .send()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;
        if !response.status().is_success() {
            return Err(status_error(response).await);
        }

        Ok(frames(response.bytes_stream()))
    }

    /// Minted per request rather than cached: it lives for a minute, and a
    /// cached one would need refreshing on a schedule to save one HMAC.
    ///
    /// Sent even with no shared secret. The signature is worthless then, but
    /// `act` is not — without it every owner-scoped route is unusable.
    fn authorization(&self, caller: &auth::Caller) -> Option<String> {
        let secret = self.secret.as_deref().unwrap_or(UNSIGNED);
        auth::token::issue_service_token(&self.audience, caller, &[""], secret)
            .ok()
            .map(|token| format!("Bearer {token}"))
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }
}

fn header_of(response: &reqwest::Response, name: reqwest::header::HeaderName) -> Option<String> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

/// Frames are delimited by a blank line and a chunk can split one anywhere, so
/// only whole frames are emitted and the remainder stays buffered.
fn frames<B: AsRef<[u8]>>(
    bytes: impl Stream<Item = reqwest::Result<B>>,
) -> impl Stream<Item = Result<Frame>> {
    let mut buffer = String::new();
    let mut pending: Vec<Result<Frame>> = Vec::new();

    Box::pin(bytes).flat_map(move |chunk| {
        match chunk {
            Err(e) => pending.push(Err(Error::Transport(e.to_string()))),
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
    Some(Frame {
        kind,
        data: serde_json::from_str(&data).unwrap_or(Value::Null),
    })
}

/// FastAPI puts its reason in `detail`; anything else is passed through
/// verbatim, since the service's own message is the only thing that says what
/// went wrong.
async fn status_error(response: reqwest::Response) -> Error {
    let code = response.status().as_u16();
    let body = response.text().await.unwrap_or_default();
    let detail = serde_json::from_str::<Value>(&body)
        .ok()
        .and_then(|v| {
            v.get("detail")
                .map(|d| d.to_string().trim_matches('"').to_string())
        })
        .unwrap_or(body);
    Error::Status { code, detail }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;

    fn chunks(pieces: &[&str]) -> impl Stream<Item = reqwest::Result<Vec<u8>>> {
        let owned: Vec<_> = pieces.iter().map(|p| Ok(p.as_bytes().to_vec())).collect();
        stream::iter(owned)
    }

    async fn collect(pieces: &[&str]) -> Vec<Frame> {
        frames(chunks(pieces))
            .filter_map(|f| async { f.ok() })
            .collect()
            .await
    }

    #[tokio::test]
    async fn a_frame_split_across_chunks_is_reassembled() {
        let seen = collect(&["event: progr", "ess\ndata: {\"step\"", ": 1}\n\n"]).await;

        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].kind, "progress");
        assert_eq!(seen[0].data["step"], 1);
    }

    #[tokio::test]
    async fn several_frames_in_one_chunk_all_arrive() {
        let seen = collect(&["event: a\ndata: 1\n\nevent: b\ndata: 2\n\n"]).await;

        let kinds: Vec<_> = seen.iter().map(|f| f.kind.as_str()).collect();
        assert_eq!(kinds, ["a", "b"]);
    }

    #[tokio::test]
    async fn a_frame_with_no_terminator_is_held_rather_than_guessed_at() {
        assert!(collect(&["event: progress\ndata: {}"]).await.is_empty());
    }

    #[tokio::test]
    async fn a_frame_with_no_event_name_is_not_a_frame() {
        assert!(collect(&["data: {}\n\n"]).await.is_empty());
    }

    #[tokio::test]
    async fn unreadable_data_still_delivers_the_event_name() {
        // The name is what a consumer switches on; losing the whole frame
        // because its payload is malformed would hide a terminal `done`.
        let seen = collect(&["event: done\ndata: not json\n\n"]).await;

        assert_eq!(seen[0].kind, "done");
        assert!(seen[0].data.is_null());
    }

    #[test]
    fn only_a_terminal_frame_ends_a_stream() {
        let final_kinds = ["done", "error", "cancelled"];
        for kind in final_kinds {
            assert!(Frame {
                kind: kind.into(),
                data: Value::Null
            }
            .is_final());
        }
        assert!(!Frame {
            kind: "progress".into(),
            data: Value::Null
        }
        .is_final());
    }

    #[test]
    fn a_token_is_addressed_to_the_service_it_was_built_for() {
        let proxy = Proxy::new("http://x", "engine", Some("s".into()));
        let header = proxy.authorization(&auth::Caller::System).expect("signed");
        let token = header.strip_prefix("Bearer ").expect("bearer scheme");

        assert!(auth::token::verify_service_token(token, "engine", "s").is_ok());
        assert!(auth::token::verify_service_token(token, "other", "s").is_err());
    }

    #[test]
    fn without_a_secret_the_claim_survives_and_the_signature_does_not() {
        let who = uuid::Uuid::new_v4();
        let proxy = Proxy::new("http://x", "engine", None);
        let header = proxy
            .authorization(&auth::Caller::User(who))
            .expect("still sent");
        let token = header.strip_prefix("Bearer ").unwrap();

        // An enforcing service refuses it, which is the intended failure.
        assert!(auth::token::verify_service_token(token, "engine", "real").is_err());
        // An open one still learns who the call is for.
        let caller = auth::token::verify_service_token(token, "engine", UNSIGNED).expect("read");
        assert!(matches!(caller, auth::Caller::User(id) if id == who));
    }

    #[test]
    fn the_token_names_the_person_the_call_is_for() {
        let who = uuid::Uuid::new_v4();
        let proxy = Proxy::new("http://x", "engine", Some("s".into()));
        let header = proxy
            .authorization(&auth::Caller::User(who))
            .expect("signed");
        let token = header.strip_prefix("Bearer ").unwrap();

        let caller = auth::token::verify_service_token(token, "engine", "s").expect("valid");
        assert!(matches!(caller, auth::Caller::User(id) if id == who));
    }
}

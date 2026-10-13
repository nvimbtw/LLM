//! Forwarding to a service.
//!
//! The backend does not model what a service owns. It authenticates a person,
//! signs a short-lived token saying who it acts for, and passes the request on
//! — so a field the service adds needs no change here, and a status it returns
//! reaches the client as itself rather than flattened into a generic failure.
//!
//! Nothing here names the LLM engine. A second service is a second `Proxy` with its
//! own audience and its own routers above it; this is the whole seam.

use futures_util::Stream;

mod sse;
#[cfg(test)]
mod tests;
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
    /// Report whether this named service event is terminal under the shared event vocabulary.
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
    /// Create a reusable service client with its audience and optional shared secret.
    /// Only connection establishment times out; durable event streams have no global request timeout.
    pub fn new(
        base_url: impl Into<String>,
        audience: impl Into<String>,
        secret: Option<String>,
    ) -> Proxy {
        Proxy {
            http: reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            base_url: base_url.into(),
            audience: audience.into(),
            secret,
        }
    }

    /// Forward an owner-scoped GET and decode the service JSON response.
    pub async fn get(&self, caller: &auth::Caller, path: &str) -> Result<Value> {
        self.call(caller, Method::GET, path, None).await
    }

    /// Forward an owner-scoped POST with an opaque service JSON body.
    pub async fn post(&self, caller: &auth::Caller, path: &str, body: Value) -> Result<Value> {
        self.call(caller, Method::POST, path, Some(body)).await
    }

    /// Forward an owner-scoped PATCH with an opaque service JSON body.
    pub async fn patch(&self, caller: &auth::Caller, path: &str, body: Value) -> Result<Value> {
        self.call(caller, Method::PATCH, path, Some(body)).await
    }

    /// Forward an owner-scoped PUT with an opaque service JSON body.
    pub async fn put(&self, caller: &auth::Caller, path: &str, body: Value) -> Result<Value> {
        self.call(caller, Method::PUT, path, Some(body)).await
    }

    /// Forward an owner-scoped DELETE and retain service status semantics.
    pub async fn delete(&self, caller: &auth::Caller, path: &str) -> Result<Value> {
        self.call(caller, Method::DELETE, path, None).await
    }

    /// Send an authenticated service request and decode its JSON, preserving non-success service statuses.
    pub async fn call(
        &self,
        caller: &auth::Caller,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value> {
        let request = self.request(caller, method, path, body);

        self.read_json(request).await
    }

    /// Send a prepared request and decode successful JSON; empty response bodies become null.
    async fn read_json(&self, request: reqwest::RequestBuilder) -> Result<Value> {
        let response = send(request).await?;

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

        let request = self.authorize(self.http.post(self.url(path)).multipart(form), caller);

        self.read_json(request).await
    }

    /// A response that is bytes rather than JSON.
    ///
    /// Buffered, not streamed: what comes back this way is capped by the
    /// service that stores it, and a streaming body would have to carry the
    /// upstream's failure through a stream this side has already answered.
    pub async fn fetch(&self, caller: &auth::Caller, path: &str) -> Result<Blob> {
        let response = send(self.request(caller, Method::GET, path, None)).await?;

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

    /// Open an authenticated event request and transfer its response body to the incremental frame decoder.
    async fn streaming(
        &self,
        caller: &auth::Caller,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<impl Stream<Item = Result<Frame>>> {
        let response = send(self.request(caller, method, path, body)).await?;

        Ok(sse::frames(response.bytes_stream()))
    }

    /// Build a signed request, adding a JSON body only when the service call supplies one.
    fn request(
        &self,
        caller: &auth::Caller,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> reqwest::RequestBuilder {
        let request = self.authorize(self.http.request(method, self.url(path)), caller);
        match body {
            Some(body) => request.json(&body),
            None => request,
        }
    }

    /// Attach the service-audience caller token to any JSON, multipart or binary request.
    fn authorize(
        &self,
        request: reqwest::RequestBuilder,
        caller: &auth::Caller,
    ) -> reqwest::RequestBuilder {
        match self.authorization(caller) {
            Some(header) => request.header(reqwest::header::AUTHORIZATION, header),
            None => request,
        }
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

    /// Resolve a service-relative path under the configured base URL without rewriting its query string.
    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }
}

/// Read an optional response header as UTF-8, omitting absent or invalid header values.
fn header_of(response: &reqwest::Response, name: reqwest::header::HeaderName) -> Option<String> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
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

/// Send one request, separating transport failures from the service's own unsuccessful status.
async fn send(request: reqwest::RequestBuilder) -> Result<reqwest::Response> {
    let response = request
        .send()
        .await
        .map_err(|error| Error::Transport(error.to_string()))?;
    if !response.status().is_success() {
        return Err(status_error(response).await);
    }
    Ok(response)
}

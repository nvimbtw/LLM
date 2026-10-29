//! Backend-owned availability monitors and bounded snapshots for administrators.

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde::Serialize;
use tokio::{sync::watch, task::JoinHandle, time::Instant};
use tokio_tungstenite::{
    connect_async_with_config,
    tungstenite::{client::IntoClientRequest, protocol::WebSocketConfig, Message},
};

use crate::Config;

const INTERVAL: Duration = Duration::from_secs(5);
const TIMEOUT: Duration = Duration::from_secs(15);
const OFFLINE_AFTER: Duration = Duration::from_secs(60);
const SEARCH_CHECK_INTERVAL: Duration = Duration::from_secs(60);

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Online,
    Reconnecting,
    Offline,
    NotConfigured,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Status {
    pub id: String,
    pub state: State,
    pub detail: Option<String>,
}

/// Owns one monitor task per configured endpoint, and aborts them when the server stops.
pub struct Monitor {
    receiver: watch::Receiver<Vec<Status>>,
    tasks: Vec<JoinHandle<()>>,
}

impl Monitor {
    /// Start monitors for four service sockets and optional backend tool endpoints.
    pub fn start(
        services: &[Config],
        bash: Option<String>,
        search: Option<String>,
        brave: bool,
    ) -> Self {
        let mut statuses: Vec<Status> = services
            .iter()
            .map(|config| Status {
                id: config.audience.clone(),
                state: State::Reconnecting,
                detail: None,
            })
            .collect();
        statuses.push(Status {
            id: "just-bash".into(),
            state: if bash.is_some() {
                State::Reconnecting
            } else {
                State::NotConfigured
            },
            detail: None,
        });
        statuses.push(Status {
            id: "searxng".into(),
            state: if search.is_some() {
                State::Reconnecting
            } else {
                State::NotConfigured
            },
            detail: if search.is_none() && brave {
                Some("Search provider: Brave".into())
            } else {
                None
            },
        });
        let (sender, receiver) = watch::channel(statuses);
        let mut tasks = Vec::new();
        for (index, config) in services.iter().cloned().enumerate() {
            tasks.push(tokio::spawn(monitor_socket(index, config, sender.clone())));
        }
        let offset = services.len();
        if let Some(url) = bash {
            tasks.push(tokio::spawn(monitor_http(
                offset,
                url,
                false,
                sender.clone(),
            )));
        }
        if let Some(url) = search {
            tasks.push(tokio::spawn(monitor_http(offset + 1, url, true, sender)));
        }
        Self { receiver, tasks }
    }

    /// Subscribe to current and subsequent full snapshots without taking monitor ownership.
    pub fn subscribe(&self) -> watch::Receiver<Vec<Status>> {
        self.receiver.clone()
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

/// Publish a changed service observation without queuing unbounded intermediate snapshots.
fn publish(
    sender: &watch::Sender<Vec<Status>>,
    index: usize,
    state: State,
    detail: Option<String>,
) {
    sender.send_if_modified(|statuses| {
        let status = &mut statuses[index];
        if status.state == state && status.detail == detail {
            return false;
        }
        status.state = state;
        status.detail = detail;
        true
    });
}

/// Classify an unsuccessful observation by the elapsed duration of the current outage.
fn unavailable(since: Instant) -> State {
    if since.elapsed() >= OFFLINE_AFTER {
        State::Offline
    } else {
        State::Reconnecting
    }
}

/// Reconnect one authenticated engine socket and update its outage state until cancelled.
async fn monitor_socket(index: usize, config: Config, sender: watch::Sender<Vec<Status>>) {
    let mut outage = Instant::now();
    loop {
        let result = socket_session(index, &config, &sender, &mut outage).await;
        if let Err(error) = result {
            eprintln!("{} health: {error}", config.audience);
        }
        publish(&sender, index, unavailable(outage), None);
        tokio::time::sleep(INTERVAL).await;
    }
}

/// Establish a signed socket and require a bounded response to every heartbeat.
async fn socket_session(
    index: usize,
    config: &Config,
    sender: &watch::Sender<Vec<Status>>,
    outage: &mut Instant,
) -> Result<(), String> {
    let url = format!("ws://{}:{}/health/socket", config.host, config.port);
    let mut request = url.into_client_request().map_err(|e| e.to_string())?;
    if let Some(secret) = &config.service_secret {
        let token =
            auth::token::issue_service_token(&config.audience, &auth::Caller::System, &[], secret)
                .map_err(|e| e.to_string())?;
        request.headers_mut().insert(
            "Authorization",
            format!("Bearer {token}").parse().map_err(
                |e: tokio_tungstenite::tungstenite::http::header::InvalidHeaderValue| e.to_string(),
            )?,
        );
    }
    let limits = WebSocketConfig::default()
        .max_message_size(Some(1024))
        .max_frame_size(Some(1024));
    let (mut socket, _) = tokio::time::timeout(
        TIMEOUT,
        connect_async_with_config(request, Some(limits), false),
    )
    .await
    .map_err(|_| "connection timed out".to_string())?
    .map_err(|e| e.to_string())?;
    loop {
        tokio::time::timeout(TIMEOUT, async {
            socket
                .send(Message::Text("ping".into()))
                .await
                .map_err(|e| e.to_string())?;
            loop {
                match socket.next().await {
                    Some(Ok(Message::Text(text))) if text == "pong" => break Ok(()),
                    Some(Ok(Message::Ping(_))) => {
                        socket.flush().await.map_err(|e| e.to_string())?
                    }
                    Some(Ok(Message::Pong(_))) => {}
                    _ => break Err("invalid heartbeat or connection closed".to_string()),
                }
            }
        })
        .await
        .map_err(|_| "heartbeat timed out".to_string())??;
        *outage = Instant::now();
        publish(sender, index, State::Online, None);
        tokio::time::sleep(INTERVAL).await;
    }
}

/// Poll a tool endpoint and periodically check SearXNG's configured JSON API.
async fn monitor_http(
    index: usize,
    base: String,
    search: bool,
    sender: watch::Sender<Vec<Status>>,
) {
    let mut outage = Instant::now();
    let client = loop {
        match reqwest::Client::builder().timeout(TIMEOUT).build() {
            Ok(client) => break client,
            Err(error) => {
                eprintln!("health HTTP client: {error}");
                publish(&sender, index, unavailable(outage), None);
                tokio::time::sleep(INTERVAL).await;
            }
        }
    };
    let mut detail = None;
    let mut next_search_check = Instant::now();
    loop {
        let url = if search {
            base.trim_end_matches('/').to_string()
        } else {
            format!("{}/health", base.trim_end_matches('/'))
        };
        let healthy = match client.get(url).send().await {
            Ok(response) if response.status().is_success() && !search => read_json(response, 4096)
                .await
                .is_some_and(|body| body.get("ok").and_then(|ok| ok.as_bool()) == Some(true)),
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        };
        if healthy {
            outage = Instant::now();
            publish(&sender, index, State::Online, detail.clone());
            if search && Instant::now() >= next_search_check {
                detail = search_api_error(&client, &base).await;
                next_search_check = Instant::now() + SEARCH_CHECK_INTERVAL;
                publish(&sender, index, State::Online, detail.clone());
            }
        } else {
            publish(&sender, index, unavailable(outage), None);
        }
        tokio::time::sleep(INTERVAL).await;
    }
}

/// Check whether the configured search API accepts JSON requests; never expose response content.
async fn search_api_error(client: &reqwest::Client, base: &str) -> Option<String> {
    let url = format!("{}/search", base.trim_end_matches('/'));
    match client
        .get(url)
        .query(&[("q", "test"), ("format", "json")])
        .send()
        .await
    {
        Ok(response) if response.status().is_success() => {
            if read_json(response, 1024 * 1024)
                .await
                .is_some_and(|body| body.get("results").is_some_and(|value| value.is_array()))
            {
                None
            } else {
                Some("Search JSON API returned an invalid response".into())
            }
        }
        Ok(response) if response.status() == reqwest::StatusCode::FORBIDDEN => {
            Some("Search JSON API refused the request; check JSON format and the limiter".into())
        }
        Ok(response) => Some(format!(
            "Search JSON API unavailable (HTTP {})",
            response.status().as_u16()
        )),
        Err(_) => Some("Search JSON API unavailable".into()),
    }
}

/// Read and parse an HTTP JSON body within a byte ceiling, rejecting truncated or oversized replies.
async fn read_json(response: reqwest::Response, limit: usize) -> Option<serde_json::Value> {
    let mut stream = response.bytes_stream();
    let mut bytes = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.ok()?;
        if bytes.len().saturating_add(chunk.len()) > limit {
            return None;
        }
        bytes.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sustained outage becomes offline, while a recent outage remains reconnecting.
    #[test]
    fn outage_states_follow_elapsed_time() {
        assert_eq!(unavailable(Instant::now()), State::Reconnecting);
        assert_eq!(unavailable(Instant::now() - OFFLINE_AFTER), State::Offline);
    }

    /// Unconfigured optional services are explicit and the selected Brave provider is identified.
    #[tokio::test]
    async fn optional_services_do_not_appear_offline() {
        let monitor = Monitor::start(&[], None, None, true);
        let snapshot = monitor.subscribe().borrow().clone();
        assert_eq!(snapshot[0].state, State::NotConfigured);
        assert_eq!(snapshot[1].state, State::NotConfigured);
        assert_eq!(
            snapshot[1].detail.as_deref(),
            Some("Search provider: Brave")
        );
    }

    /// The configured sandbox is online only after its health endpoint returns the expected JSON.
    #[tokio::test]
    async fn sandbox_probe_checks_the_response_body() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 2048];
            let size = stream.read(&mut request).await.unwrap();
            assert!(String::from_utf8_lossy(&request[..size]).starts_with("GET /health "));
            stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{\"ok\":true}").await.unwrap();
        });
        let monitor = Monitor::start(&[], Some(format!("http://{address}")), None, false);
        let mut snapshots = monitor.subscribe();
        tokio::time::timeout(Duration::from_secs(2), async {
            while snapshots.borrow_and_update()[0].state != State::Online {
                snapshots.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        server.await.unwrap();
    }

    /// A refused search API supplies a configuration hint without exposing its response body.
    #[tokio::test]
    async fn search_refusal_reports_how_to_recover() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0u8; 2048];
            let size = stream.read(&mut request).await.unwrap();
            assert!(String::from_utf8_lossy(&request[..size]).starts_with("GET /search?"));
            stream
                .write_all(
                    b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
        });
        let detail = search_api_error(&reqwest::Client::new(), &format!("http://{address}"))
            .await
            .unwrap();
        assert!(detail.contains("JSON format") && detail.contains("limiter"));
        server.await.unwrap();
    }

    /// A signed heartbeat survives control frames, detects disconnect and reconnects to a replacement.
    /// Tungstenite's handshake callback fixes the error type as a large HTTP response.
    #[allow(clippy::result_large_err)]
    #[tokio::test]
    async fn signed_socket_marks_online_and_detects_disconnect() {
        use tokio::net::TcpListener;
        use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = tokio::spawn(async move {
            for _ in 0..2 {
                let (stream, _) = listener.accept().await.unwrap();
                let mut socket = tokio_tungstenite::accept_hdr_async(
                    stream,
                    |request: &Request, response: Response| {
                        let token = request.headers()["Authorization"]
                            .to_str()
                            .unwrap()
                            .strip_prefix("Bearer ")
                            .unwrap();
                        assert!(
                            auth::token::verify_service_token(token, "llm-engine", "secret")
                                .is_ok()
                        );
                        Ok(response)
                    },
                )
                .await
                .unwrap();
                assert_eq!(
                    socket.next().await.unwrap().unwrap().into_text().unwrap(),
                    "ping"
                );
                socket.send(Message::Ping("probe".into())).await.unwrap();
                socket.send(Message::Text("pong".into())).await.unwrap();
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        });
        let config = Config {
            audience: "llm-engine".into(),
            python: "unused".into(),
            dir: ".".into(),
            host: "127.0.0.1".into(),
            port,
            autostart: false,
            service_secret: Some("secret".into()),
            secret_var: "UNUSED".into(),
            extra_env: vec![],
            startup_timeout: TIMEOUT,
        };
        let monitor = Monitor::start(&[config], None, None, false);
        let mut snapshots = monitor.subscribe();
        tokio::time::timeout(Duration::from_secs(2), async {
            while snapshots.borrow_and_update()[0].state != State::Online {
                snapshots.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        tokio::time::timeout(Duration::from_secs(8), async {
            while snapshots.borrow_and_update()[0].state == State::Online {
                snapshots.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        assert_eq!(snapshots.borrow()[0].state, State::Reconnecting);
        tokio::time::timeout(Duration::from_secs(8), async {
            while snapshots.borrow_and_update()[0].state != State::Online {
                snapshots.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        server.await.unwrap();
    }
}

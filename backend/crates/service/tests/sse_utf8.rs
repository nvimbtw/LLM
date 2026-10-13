//! The public service proxy must preserve UTF-8 across real HTTP body chunks.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Duration;

use futures_util::StreamExt;
use serde_json::json;

/// Serve two deliberately separated HTTP chunks and return the one-request server URL and owner.
/// Read/write deadlines bound the stub; its join handle surfaces unexpected server failures.
fn chunked_server(first: Vec<u8>, second: Vec<u8>) -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind test stub");
    let url = format!("http://{}", listener.local_addr().expect("stub address"));
    let owner = std::thread::spawn(move || {
        let (mut socket, _) = listener.accept().expect("accept proxy request");
        socket
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        socket
            .set_write_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut request = [0; 8192];
        let received = socket.read(&mut request).expect("read request");
        assert!(received > 0, "proxy sent an empty request");
        socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n").unwrap();
        write_chunk(&mut socket, &first);
        std::thread::sleep(Duration::from_millis(100));
        write_chunk(&mut socket, &second);
        socket.write_all(b"0\r\n\r\n").unwrap();
    });
    (url, owner)
}

/// Write one HTTP chunk verbatim and flush so the proxy can observe its incomplete UTF-8 suffix.
fn write_chunk(socket: &mut std::net::TcpStream, bytes: &[u8]) {
    write!(socket, "{:x}\r\n", bytes.len()).unwrap();
    socket.write_all(bytes).unwrap();
    socket.write_all(b"\r\n").unwrap();
    socket.flush().unwrap();
}

/// Verify the public HTTP adapter emits multilingual token text unchanged after a mid-character split.
#[tokio::test]
async fn a_multibyte_token_survives_real_http_chunk_boundaries() {
    let event = "event: token\ndata: {\"token\":\"ž漢🦀\"}\n\n";
    let split = event.find('ž').unwrap() + 1;
    let (url, owner) = chunked_server(
        event.as_bytes()[..split].to_vec(),
        event.as_bytes()[split..].to_vec(),
    );
    let proxy = service::Proxy::new(url, "stub", None);
    let stream = proxy
        .events(&auth::Caller::System, "/events")
        .await
        .expect("open event stream");
    let frames: Vec<_> = stream.collect().await;
    owner.join().expect("stub completed");
    assert_eq!(frames.len(), 1);
    let frame = frames[0].as_ref().expect("valid event");
    assert_eq!(frame.kind, "token");
    assert_eq!(frame.data, json!({"token":"ž漢🦀"}));
}

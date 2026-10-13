//! Incremental service events; frame bytes own partial UTF-8 until the event is complete.

use super::Frame;
use crate::{Error, Result};
use futures_util::{Stream, StreamExt};
use serde_json::Value;

#[derive(Default)]
struct Decoder {
    pending_bytes: Vec<u8>,
}

impl Decoder {
    /// Buffer a network chunk and emit every complete LF-delimited event, retaining an unfinished suffix.
    /// Decode UTF-8 only after framing so splitting a character across network reads cannot corrupt it.
    fn push(&mut self, chunk: &[u8]) -> Vec<Frame> {
        self.pending_bytes.extend_from_slice(chunk);
        let mut frames = Vec::new();
        let mut consumed = 0;
        while let Some(split) = self.pending_bytes[consumed..]
            .windows(2)
            .position(|bytes| bytes == b"\n\n")
        {
            let end = consumed + split;
            if let Some(frame) = parse(&String::from_utf8_lossy(&self.pending_bytes[consumed..end]))
            {
                frames.push(frame);
            }
            consumed = end + 2;
        }
        self.pending_bytes.drain(..consumed);
        frames
    }
}

/// Convert a byte stream to frames while preserving transport errors and retaining incomplete trailing events.
pub(super) fn frames<B: AsRef<[u8]>>(
    bytes: impl Stream<Item = reqwest::Result<B>>,
) -> impl Stream<Item = Result<Frame>> {
    let mut decoder = Decoder::default();
    Box::pin(bytes).flat_map(move |chunk| {
        let frames = match chunk {
            Ok(chunk) => decoder.push(chunk.as_ref()).into_iter().map(Ok).collect(),
            Err(error) => vec![Err(Error::Transport(error.to_string()))],
        };
        futures_util::stream::iter(frames)
    })
}

/// Parse the established service SSE subset: named events and concatenated `data: ` lines containing JSON.
/// Malformed JSON becomes null, unnamed events are ignored, and unknown event names are preserved.
fn parse(frame: &str) -> Option<Frame> {
    let mut kind = String::new();
    let mut data = String::new();
    for line in frame.lines() {
        if let Some(rest) = line.strip_prefix("event: ") {
            kind = rest.trim().to_owned();
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every split inside multilingual UTF-8 text produces the same lossless event as a single network read.
    #[test]
    fn utf8_is_lossless_at_every_network_split() {
        let event = "event: token\ndata: {\"token\":\"ž漢🦀\"}\n\n".as_bytes();
        for split in 0..=event.len() {
            let mut decoder = Decoder::default();
            let mut frames = decoder.push(&event[..split]);
            frames.extend(decoder.push(&event[split..]));
            assert_eq!(frames.len(), 1, "split {split}");
            assert_eq!(frames[0].data["token"], "ž漢🦀", "split {split}");
        }
    }

    /// One-byte reads preserve adjacent event boundaries and multibyte characters together.
    #[test]
    fn bytewise_reads_preserve_multiple_frames() {
        let mut decoder = Decoder::default();
        let frames: Vec<_> = "event: token\ndata: \"ž\"\n\nevent: done\ndata: {}\n\n"
            .as_bytes()
            .iter()
            .flat_map(|byte| decoder.push(&[*byte]))
            .collect();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].data, "ž");
        assert_eq!(frames[1].kind, "done");
    }

    /// Retain the existing data-line concatenation contract instead of silently introducing a new SSE dialect.
    #[test]
    fn data_lines_keep_legacy_concatenation() {
        let mut decoder = Decoder::default();
        let frames = decoder.push(b"event: progress\ndata: {\"step\":\ndata: 2}\n\n");
        assert_eq!(frames[0].data["step"], 2);
    }

    /// The service protocol still requires LF frame terminators; CRLF and unfinished events remain buffered.
    #[test]
    fn unsupported_crlf_and_truncated_events_are_not_emitted() {
        assert!(Decoder::default()
            .push(b"event: done\r\ndata: {}\r\n\r\n")
            .is_empty());
        assert!(Decoder::default()
            .push(b"event: token\ndata: \"\xc5")
            .is_empty());
    }

    /// A transport error remains visible and does not discard the bytes of an event already in progress.
    #[tokio::test]
    async fn transport_errors_remain_visible_without_losing_pending_bytes() {
        let error = reqwest::Client::new().get("not a url").build().unwrap_err();
        let chunks: Vec<reqwest::Result<Vec<u8>>> = vec![
            Ok(b"event: done\n".to_vec()),
            Err(error),
            Ok(b"data: {}\n\n".to_vec()),
        ];
        let seen: Vec<_> = frames(futures_util::stream::iter(chunks)).collect().await;
        assert!(matches!(&seen[0], Err(Error::Transport(_))));
        assert_eq!(seen[1].as_ref().unwrap().kind, "done");
    }
}

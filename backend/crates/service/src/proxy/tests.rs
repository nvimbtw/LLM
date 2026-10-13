use super::sse::frames;
use super::*;
use futures_util::{stream, StreamExt};

/// Build deterministic byte chunks for exercising the stream adapter.
fn chunks(pieces: &[&str]) -> impl Stream<Item = reqwest::Result<Vec<u8>>> {
    let owned: Vec<_> = pieces.iter().map(|p| Ok(p.as_bytes().to_vec())).collect();
    stream::iter(owned)
}

/// Collect successfully parsed frames from a fixed chunk script.
async fn collect(pieces: &[&str]) -> Vec<Frame> {
    frames(chunks(pieces))
        .filter_map(|f| async { f.ok() })
        .collect()
        .await
}

#[tokio::test]
/// Chunk boundaries do not change complete ASCII event payloads.
async fn a_frame_split_across_chunks_is_reassembled() {
    let seen = collect(&["event: progr", "ess\ndata: {\"step\"", ": 1}\n\n"]).await;

    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].kind, "progress");
    assert_eq!(seen[0].data["step"], 1);
}

#[tokio::test]
/// All complete events in one network chunk remain ordered.
async fn several_frames_in_one_chunk_all_arrive() {
    let seen = collect(&["event: a\ndata: 1\n\nevent: b\ndata: 2\n\n"]).await;

    let kinds: Vec<_> = seen.iter().map(|f| f.kind.as_str()).collect();
    assert_eq!(kinds, ["a", "b"]);
}

#[tokio::test]
/// An unfinished event is not invented at end of stream.
async fn a_frame_with_no_terminator_is_held_rather_than_guessed_at() {
    assert!(collect(&["event: progress\ndata: {}"]).await.is_empty());
}

#[tokio::test]
/// Data without an event name remains ignored.
async fn a_frame_with_no_event_name_is_not_a_frame() {
    assert!(collect(&["data: {}\n\n"]).await.is_empty());
}

#[tokio::test]
/// Malformed JSON preserves terminal event identity with a null payload.
async fn unreadable_data_still_delivers_the_event_name() {
    let seen = collect(&["event: done\ndata: not json\n\n"]).await;

    assert_eq!(seen[0].kind, "done");
    assert!(seen[0].data.is_null());
}

#[test]
/// Only the documented terminal event names end service event observation.
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
/// Each outgoing token is valid only for its configured service audience.
fn a_token_is_addressed_to_the_service_it_was_built_for() {
    let proxy = Proxy::new("http://x", "llm-engine", Some("s".into()));
    let header = proxy.authorization(&auth::Caller::System).expect("signed");
    let token = header.strip_prefix("Bearer ").expect("bearer scheme");

    assert!(auth::token::verify_service_token(token, "llm-engine", "s").is_ok());
    assert!(auth::token::verify_service_token(token, "other", "s").is_err());
}

#[test]
/// Development tokens preserve caller identity while enforcing services reject the unsigned carrier.
fn without_a_secret_the_claim_survives_and_the_signature_does_not() {
    let who = uuid::Uuid::new_v4();
    let proxy = Proxy::new("http://x", "llm-engine", None);
    let header = proxy
        .authorization(&auth::Caller::User(who))
        .expect("still sent");
    let token = header.strip_prefix("Bearer ").unwrap();

    assert!(auth::token::verify_service_token(token, "llm-engine", "real").is_err());
    let caller = auth::token::verify_service_token(token, "llm-engine", UNSIGNED).expect("read");
    assert!(matches!(caller, auth::Caller::User(id) if id == who));
}

#[test]
/// A user-scoped service token retains the exact authenticated user identity.
fn the_token_names_the_person_the_call_is_for() {
    let who = uuid::Uuid::new_v4();
    let proxy = Proxy::new("http://x", "llm-engine", Some("s".into()));
    let header = proxy
        .authorization(&auth::Caller::User(who))
        .expect("signed");
    let token = header.strip_prefix("Bearer ").unwrap();

    let caller = auth::token::verify_service_token(token, "llm-engine", "s").expect("valid");
    assert!(matches!(caller, auth::Caller::User(id) if id == who));
}

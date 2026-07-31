use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::chat_template::{build_prompt, Turn};
use crate::training::infer_session::InferSession;
use crate::training::transformer::Model;
use tokenize::pretokenize::lookup_special;

// A loaded model, ready to generate: the tokenizer merges plus the device-resident
// session that holds the weights.
//
// The host-side `Model` is deliberately **not** kept. `InferSession::new` uploads
// every weight to the device, and nothing on this path reads them from the host
// again — holding the `Vec<Vec<f32>>` copy would be tens of megabytes of resident
// host memory serving no reader.
struct CachedModel {
    name: String,
    session: InferSession,
    merges: Vec<((u32, u32), u32)>,
}

static MODEL_CACHE: Mutex<Option<CachedModel>> = Mutex::new(None);

pub enum Marker {
    EndOfText,
    Code,
    EndCode,
    User,
    EndUser,
    Assistant,
    EndAssistant,
    System,
    EndSystem,
    Unknown(String),
}

/// Scans decoded text for any special marker tokens and returns them in order of appearance.
pub fn find_markers(text: &str) -> Vec<Marker> {
    let mut markers = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if let Some(marker_str) = lookup_special(chars[i]) {
            markers.push(marker_from_str(marker_str));
        }
        i += 1;
    }
    markers
}

fn marker_from_str(s: &str) -> Marker {
    match s {
        "<|endoftext|>" => Marker::EndOfText,
        "<|code|>" => Marker::Code,
        "<|/code|>" => Marker::EndCode,
        "<|user|>" => Marker::User,
        "<|/user|>" => Marker::EndUser,
        "<|assistant|>" => Marker::Assistant,
        "<|/assistant|>" => Marker::EndAssistant,
        "<|system|>" => Marker::System,
        "<|/system|>" => Marker::EndSystem,
        _ => Marker::Unknown(s.to_string()),
    }
}

// The result of one generation: the decoded reply plus token counts, used for
// usage tracking. input_tokens is the prompt length after BPE encoding;
// output_tokens is how many tokens the model actually generated.
pub struct GenerationOutput {
    pub text: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

// Token length of `text` under a model's BPE merges. Loads only the merges
// (a small file) — not the model weights — so it's cheap enough to call on the
// async executor when persisting a message. Used to record per-message token
// counts for the chat's cumulative usage display.
pub fn encode_count(model_dir: &Path, text: &str) -> Result<usize, String> {
    let merges_path = model_dir.join("merges.bin");
    let merges =
        tokenize::bpe::load_merges(&merges_path.to_string_lossy()).map_err(|e| e.to_string())?;
    Ok(tokenize::bpe::encode(text, &merges).len())
}

pub fn generate_response(
    model_dir: &Path,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: u32,
) -> Result<String, String> {
    let out = generate_response_streaming(model_dir, prompt, max_tokens, temperature, top_k, |_| {})?;
    Ok(out.text)
}

pub fn generate_response_streaming<F: FnMut(&str)>(
    model_dir: &Path,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: u32,
    on_token: F,
) -> Result<GenerationOutput, String> {
    ensure_loaded(model_dir)?;
    let mut cache = MODEL_CACHE.lock().map_err(|e| e.to_string())?;
    let c = cache.as_mut().unwrap();

    let seq = tokenize::bpe::encode(prompt, &c.merges);
    let input_tokens = seq.len();
    // The bare-prompt path has no job behind it to cancel; only the chat path,
    // which runs as a queued job, passes a flag.
    Ok(stream_generation(
        &mut c.session,
        &c.merges,
        seq,
        input_tokens,
        max_tokens,
        temperature,
        top_k,
        None,
        on_token,
    ))
}

// Multi-turn chat generation: renders the whole conversation (system + history)
// through the canonical template, truncating oldest-first to the loaded model's
// context window, then generates the next assistant turn. This is what the chat
// route uses so the model actually sees the conversation, not just the last line.
// `cancel`, when supplied, is checked once per generated token: a cancelled turn
// returns the text produced so far rather than erroring, since a partial reply is
// still a real thing the user watched arrive and the caller decides what to do
// with it.
#[allow(clippy::too_many_arguments)]
pub fn generate_chat_response_streaming<F: FnMut(&str)>(
    model_dir: &Path,
    system: Option<&str>,
    history: &[Turn],
    max_tokens: usize,
    temperature: f32,
    top_k: u32,
    cancel: Option<&AtomicBool>,
    on_token: F,
) -> Result<GenerationOutput, String> {
    ensure_loaded(model_dir)?;
    let mut cache = MODEL_CACHE.lock().map_err(|e| e.to_string())?;
    let c = cache.as_mut().unwrap();

    // The session was built at the model's context window, which is the real one.
    let ctx_window = c.session.ctx_window();
    // Leave headroom so the model can generate a reply before the sliding window
    // starts evicting the freshest prompt tokens.
    let reserve = (ctx_window / 4).max(8);
    let seq = build_prompt(system, history, &c.merges, ctx_window, reserve);
    let input_tokens = seq.len();
    Ok(stream_generation(
        &mut c.session,
        &c.merges,
        seq,
        input_tokens,
        max_tokens,
        temperature,
        top_k,
        cancel,
        on_token,
    ))
}

// Load the model + merges into the cache if a different model is requested. The
// caller (backend) builds model_dir via its storage module; the tokenizer merges
// live alongside the weights, written there at train time.
fn ensure_loaded(model_dir: &Path) -> Result<(), String> {
    let model_dir_str = model_dir.to_string_lossy().to_string();
    let merges_path = model_dir.join("merges.bin");
    let mut cache = MODEL_CACHE.lock().map_err(|e| e.to_string())?;
    if cache.as_ref().map(|c| c.name.as_str()) != Some(model_dir_str.as_str()) {
        let merges = tokenize::bpe::load_merges(&merges_path.to_string_lossy())
            .map_err(|e| e.to_string())?;
        let model = Model::load(&model_dir_str).map_err(|e| e.to_string())?;
        // Drop the previous session before building the new one so the two models'
        // weights are never both resident on an 8 GB card.
        *cache = None;
        let session = InferSession::new(&model, model.ctx_window());
        *cache = Some(CachedModel {
            name: model_dir_str.clone(),
            session,
            merges,
        });
    }
    Ok(())
}

/// Drop the cached model and free its device memory.
///
/// There is one GPU and one queue, so a resident inference session and a training
/// run want the same card. The session's footprint is modest (the weights in f16
/// plus scratch at the context window), but it is not nothing, and the shared
/// `attn`/`ff` scratch it sizes at `ctx` is the same file-static arena a training
/// `Workspace` uses. The queue calls this before starting a job that needs the
/// device for something else, so that ownership is explicit rather than incidental.
///
/// The next generation call reloads from disk, which is the intended cost.
pub fn release() {
    if let Ok(mut cache) = MODEL_CACHE.lock() {
        *cache = None;
    }
}

// The core autoregressive loop, shared by the single-prompt and chat entry
// points. Takes the already-tokenized prompt `seq` and streams decoded token
// text through `on_token`, stopping at the first end-of-turn marker.
#[allow(clippy::too_many_arguments)]
fn stream_generation<F: FnMut(&str)>(
    session: &mut InferSession,
    merges: &[((u32, u32), u32)],
    mut seq: Vec<u32>,
    input_tokens: usize,
    max_tokens: usize,
    temperature: f32,
    top_k: u32,
    cancel: Option<&AtomicBool>,
    mut on_token: F,
) -> GenerationOutput {
    let mut output_ids: Vec<u32> = Vec::new();
    let mut response = String::new();
    // Indexed once, not once per token: the free `bpe::decode` rebuilds its merge
    // table on every call, which measured 292 us — 15% of a generated token — for a
    // table that never changes.
    let decoder = tokenize::bpe::Decoder::new(merges);

    for _ in 0..max_tokens {
        // Checked before the forward pass rather than after, so a cancel lands
        // at the cost of at most one token rather than one more full pass.
        if cancel.is_some_and(|c| c.load(Ordering::SeqCst)) {
            break;
        }
        // The session truncates to its context window internally, so a sequence
        // that has outgrown the window is handled there rather than here.
        let logits = session.logits_last(&seq);
        if logits.is_empty() {
            break;
        }

        let next_token =
            crate::sample::sample(&logits, temperature, top_k as usize, &mut rand::thread_rng());
        output_ids.push(next_token);
        seq.push(next_token);

        let new_response = decoder.decode(&output_ids);
        if new_response.len() > response.len() {
            let token_text = &new_response[response.len()..];
            on_token(token_text);
            response = new_response;
        }

        if find_stop(&response).is_some() {
            break;
        }
    }

    // The reply is everything up to the first end-of-turn marker; the marker
    // itself is dropped from the returned/persisted text.
    let text = match find_stop(&response) {
        Some(pos) => response[..pos].to_string(),
        None => response,
    };

    GenerationOutput {
        text,
        input_tokens,
        output_tokens: output_ids.len(),
    }
}

// Generation halts at the first end-of-turn marker: `<|/assistant|>` (the model
// finished its turn) or `<|endoftext|>`. Returns the byte offset of the earliest
// one, or None if neither has appeared yet.
fn find_stop(text: &str) -> Option<usize> {
    let eot = text.find("<|endoftext|>");
    let eoa = text.find("<|/assistant|>");
    match (eot, eoa) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── marker_from_str ──

    #[test]
    fn test_marker_from_str_endoftext() {
        assert!(matches!(marker_from_str("<|endoftext|>"), Marker::EndOfText));
    }

    #[test]
    fn test_marker_from_str_code() {
        assert!(matches!(marker_from_str("<|code|>"), Marker::Code));
        assert!(matches!(marker_from_str("<|/code|>"), Marker::EndCode));
    }

    #[test]
    fn test_marker_from_str_user() {
        assert!(matches!(marker_from_str("<|user|>"), Marker::User));
        assert!(matches!(marker_from_str("<|/user|>"), Marker::EndUser));
    }

    #[test]
    fn test_marker_from_str_assistant() {
        assert!(matches!(marker_from_str("<|assistant|>"), Marker::Assistant));
        assert!(matches!(marker_from_str("<|/assistant|>"), Marker::EndAssistant));
    }

    #[test]
    fn test_marker_from_str_system() {
        assert!(matches!(marker_from_str("<|system|>"), Marker::System));
        assert!(matches!(marker_from_str("<|/system|>"), Marker::EndSystem));
    }

    #[test]
    fn test_marker_from_str_unknown() {
        assert!(matches!(marker_from_str("<|unknown|>"), Marker::Unknown(_)));
        if let Marker::Unknown(s) = marker_from_str("<|custom|>") {
            assert_eq!(s, "<|custom|>");
        } else {
            panic!("expected Unknown variant");
        }
    }

    // ── find_markers ──

    #[test]
    fn test_find_markers_empty() {
        assert!(find_markers("").is_empty());
    }

    #[test]
    fn test_find_markers_no_markers() {
        assert!(find_markers("hello world, nothing special here!").is_empty());
    }

    #[test]
    fn test_find_markers_with_pua_chars() {
        let text = "\u{E000}Hello\u{E004}World\u{E006}!";
        let markers = find_markers(text);
        assert_eq!(markers.len(), 3);
        assert!(matches!(markers[0], Marker::EndOfText));
        assert!(matches!(markers[1], Marker::User));
        assert!(matches!(markers[2], Marker::Assistant));
    }

    #[test]
    fn test_find_markers_adjacent() {
        let text = "\u{E000}\u{E004}";
        let markers = find_markers(text);
        assert_eq!(markers.len(), 2);
        assert!(matches!(markers[0], Marker::EndOfText));
        assert!(matches!(markers[1], Marker::User));
    }
}

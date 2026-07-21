use std::path::Path;
use std::sync::Mutex;

use crate::chat_template::{build_prompt, Turn};
use crate::training::transformer::Model;
use tokenize::pretokenize::lookup_special;

struct CachedModel {
    name: String,
    model: Model,
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
    let cache = MODEL_CACHE.lock().map_err(|e| e.to_string())?;
    let c = cache.as_ref().unwrap();

    let seq = tokenize::bpe::encode(prompt, &c.merges);
    let input_tokens = seq.len();
    Ok(stream_generation(
        &c.model, &c.merges, seq, input_tokens, max_tokens, temperature, top_k, on_token,
    ))
}

// Multi-turn chat generation: renders the whole conversation (system + history)
// through the canonical template, truncating oldest-first to the loaded model's
// context window, then generates the next assistant turn. This is what the chat
// route uses so the model actually sees the conversation, not just the last line.
pub fn generate_chat_response_streaming<F: FnMut(&str)>(
    model_dir: &Path,
    system: Option<&str>,
    history: &[Turn],
    max_tokens: usize,
    temperature: f32,
    top_k: u32,
    on_token: F,
) -> Result<GenerationOutput, String> {
    ensure_loaded(model_dir)?;
    let cache = MODEL_CACHE.lock().map_err(|e| e.to_string())?;
    let c = cache.as_ref().unwrap();

    // The model's positional table defines its real context window.
    let ctx_window = c.model.positional_table.len();
    // Leave headroom so the model can generate a reply before the sliding window
    // starts evicting the freshest prompt tokens.
    let reserve = (ctx_window / 4).max(8);
    let seq = build_prompt(system, history, &c.merges, ctx_window, reserve);
    let input_tokens = seq.len();
    Ok(stream_generation(
        &c.model, &c.merges, seq, input_tokens, max_tokens, temperature, top_k, on_token,
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
        *cache = Some(CachedModel {
            name: model_dir_str.clone(),
            model,
            merges,
        });
    }
    Ok(())
}

// The core autoregressive loop, shared by the single-prompt and chat entry
// points. Takes the already-tokenized prompt `seq` and streams decoded token
// text through `on_token`, stopping at the first end-of-turn marker.
#[allow(clippy::too_many_arguments)]
fn stream_generation<F: FnMut(&str)>(
    model: &Model,
    merges: &[((u32, u32), u32)],
    mut seq: Vec<u32>,
    input_tokens: usize,
    max_tokens: usize,
    temperature: f32,
    top_k: u32,
    mut on_token: F,
) -> GenerationOutput {
    pollster::block_on(async {
        // A fixed value larger than the positional table would overrun it in
        // forward_seq (the sliding window keeps more tokens than there are
        // positions for), so the model's own table length is the ceiling.
        let ctx_window = model.positional_table.len();
        let mut output_ids: Vec<u32> = Vec::new();
        let mut response = String::new();

        for _ in 0..max_tokens {
            let hidden = model.forward_seq(&seq, model.layers as usize, ctx_window).await;
            if hidden.is_empty() {
                break;
            }
            let last_hidden = vec![hidden.last().unwrap().clone()];
            let logits = model.unembed(&last_hidden).await;

            let next_token = sample_token(&logits[0], temperature, top_k);
            output_ids.push(next_token);
            seq.push(next_token);

            let new_response = tokenize::bpe::decode(&output_ids, merges);
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
    })
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

fn sample_token(logits: &[f32], temperature: f32, top_k: u32) -> u32 {
    let mut candidates: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &p)| (i as u32, p))
        .collect();

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(top_k as usize);

    let total: f32 = candidates
        .iter()
        .map(|(_, p)| (p / temperature).exp())
        .sum();
    let mut r = rand::random::<f32>() * total;
    let mut next_token = candidates[0].0;
    for &(tok, prob) in &candidates {
        r -= (prob / temperature).exp();
        if r <= 0.0 {
            next_token = tok;
            break;
        }
    }
    next_token
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

    // ── sample_token ──

    #[test]
    fn test_sample_token_returns_valid_index() {
        let logits = vec![1.0, 2.0, 3.0];
        for _ in 0..20 {
            let token = sample_token(&logits, 1.0, 3);
            assert!(token < 3, "token {} out of bounds", token);
        }
    }

    #[test]
    fn test_sample_token_top_k_limits() {
        let logits = vec![10.0, 9.0, 1.0, 0.5, 0.3];
        for _ in 0..100 {
            let token = sample_token(&logits, 1.0, 2);
            assert!(
                token < 2,
                "top_k=2 should only return tokens 0 or 1, got {}",
                token
            );
        }
    }

    #[test]
    fn test_sample_token_top_k_larger_than_vocab() {
        let logits = vec![0.1, 0.2, 0.3];
        for _ in 0..20 {
            let token = sample_token(&logits, 1.0, 100);
            assert!(token < 3, "token {} out of bounds", token);
        }
    }

    #[test]
    fn test_sample_token_negative_logits() {
        let logits = vec![-1.0, -2.0, -3.0, -4.0];
        for _ in 0..20 {
            let token = sample_token(&logits, 1.0, 4);
            assert!(token < 4, "token {} out of bounds", token);
        }
    }

    #[test]
    fn test_sample_token_temperature_zero_picks_highest() {
        let logits = vec![0.1, 0.8, 0.3, 0.05];
        for _ in 0..50 {
            assert_eq!(
                sample_token(&logits, 0.0, 4),
                1,
                "temperature=0 should always pick the highest logit (index 1)"
            );
        }
    }

    #[test]
    fn test_sample_token_single_element() {
        let logits = vec![42.0];
        for _ in 0..10 {
            assert_eq!(sample_token(&logits, 1.0, 1), 0);
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

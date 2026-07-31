// The teacher: an off-the-shelf instruction model, run locally on the GPU, whose
// only job is to emit text that `synth` turns into training data.
//
// This crate is deliberately thin and knows nothing about SFT. It loads a GGUF,
// renders a chat prompt with the model's *own* template, samples tokens, and
// hands back a string. Every decision about what to ask for, what counts as
// valid, and where it gets written belongs to `synth` — which is also why this
// crate has no `serde_json`, no corpus paths, and no notion of a conversation.
//
// **Why llama.cpp and not a Rust tensor framework.** `llm` is this project's own
// CUDA transformer, and it exists to train *one* architecture. The teacher is a
// different model family (Qwen/Gemma), quantized, with its own tokenizer and
// chat template — none of which `llm` can load. The alternative was a second
// tensor framework (candle) sitting beside the hand-written kernels, competing
// for the same VRAM with a second allocator and a second CUDA context. Binding a
// C++ library through FFI is the pattern `llm` already uses (`build.rs` → nvcc →
// `extern "C"`), so this adds a dependency, not a paradigm.
//
// **This holds the GPU.** A 4B model at Q4_K_M is ~2.4 GB of VRAM plus its KV
// cache, on a device with 8 GB that a training run also wants. So work driven
// through here belongs on `db::job::JobLane::Device` and must take the single GPU
// line, exactly like a training run or a benchmark. Nothing in this crate
// enforces that — it cannot see the queue — but a caller that ignores it will
// OOM the device rather than merely run slowly.

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::TokenToStringError;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use serde::{Deserialize, Serialize};

// ── configuration ───────────────────────────────────────────────────────────

/// Which local model to run, and how much of the device to give it.
///
/// Persisted as part of a job's `config`, so these are the knobs a retry
/// replays. The model is named by *file name*, never a path: it is resolved
/// inside `storage::teacher_dir()` so a run cannot be pointed at an arbitrary
/// file on disk from the browser.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherConfig {
    pub model_file: String,
    /// Context window to allocate. This is the binding VRAM cost after the
    /// weights themselves, and the reason it is configurable: the KV cache
    /// scales linearly with it, so a window big enough for a long prompt plus
    /// three escalating answers is a real allocation, not a free upper bound.
    #[serde(default = "default_n_ctx")]
    pub n_ctx: u32,
    /// Layers to offload to the GPU. The default is deliberately larger than any
    /// small model has: llama.cpp clamps it to the layer count, so "more than
    /// all of them" is the portable way to say "put the whole model on the
    /// device". Lower it only to fit a model that otherwise would not.
    #[serde(default = "default_n_gpu_layers")]
    pub n_gpu_layers: u32,
    /// CPU threads for whatever stays on the host. Irrelevant to fully
    /// offloaded layers.
    #[serde(default = "default_n_threads")]
    pub n_threads: i32,
}

// Raised from 8192 after the first real run produced no long answers at all. Three
// escalating thinking blocks plus three replies is on the order of 4-5k generated
// tokens, and the prompt (exemplars plus instructions) is another ~3.5k — at 8192
// the generation budget was clamped to what was left of the window, and since the
// answers come out shortest-first, the one that got cut was always the long one.
//
// The cost is VRAM: the KV cache scales linearly, so this is roughly 1.8 GB for a 4B
// at 12288 against ~1.2 GB at 8192. Still comfortable next to 2.5 GB of weights on
// an 8 GB card, and cheaper than a tier that never fills.
fn default_n_ctx() -> u32 {
    12288
}
fn default_n_gpu_layers() -> u32 {
    999
}
fn default_n_threads() -> i32 {
    4
}

impl Default for TeacherConfig {
    fn default() -> Self {
        TeacherConfig {
            model_file: String::new(),
            n_ctx: default_n_ctx(),
            n_gpu_layers: default_n_gpu_layers(),
            n_threads: default_n_threads(),
        }
    }
}

/// How to sample. Separate from `TeacherConfig` because the model is loaded once
/// and sampled many times, sometimes with different settings per call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleOpts {
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub top_k: i32,
    /// Hard cap on generated tokens. Reaching it is reported as
    /// `Generated::truncated`, which matters: a truncated response has lost
    /// whatever it was about to say, and for delimiter-formatted output that
    /// usually means the final record is unparseable.
    pub max_tokens: usize,
    pub seed: u32,
    /// DRY repetition penalty. A small local model asked for three answers to
    /// one question will otherwise write the second and third as near-copies of
    /// the first — the exact failure this pipeline exists to avoid — so it is on
    /// by default rather than offered as a knob. `0.0` disables it.
    #[serde(default = "default_dry_multiplier")]
    pub dry_multiplier: f32,
    #[serde(default = "default_dry_base")]
    pub dry_base: f32,
    #[serde(default = "default_dry_allowed_length")]
    pub dry_allowed_length: i32,
}

fn default_dry_multiplier() -> f32 {
    0.8
}
fn default_dry_base() -> f32 {
    1.75
}
fn default_dry_allowed_length() -> i32 {
    4
}

impl Default for SampleOpts {
    fn default() -> Self {
        SampleOpts {
            temperature: 1.0,
            top_p: 0.95,
            min_p: 0.05,
            top_k: 60,
            // Has to cover three thinking blocks and three replies in one response.
            // The long block alone targets 320-500 words; 4096 total left nothing for
            // it, and truncation costs whichever answer is last.
            max_tokens: 6144,
            seed: 0,
            dry_multiplier: default_dry_multiplier(),
            dry_base: default_dry_base(),
            dry_allowed_length: default_dry_allowed_length(),
        }
    }
}

// ── what a call produces ────────────────────────────────────────────────────

/// One completion, plus the facts the caller needs to judge it.
#[derive(Debug, Clone)]
pub struct Generated {
    pub text: String,
    pub tokens: usize,
    /// Stopped on the token budget (or the end of the window) rather than on the
    /// model's own end-of-generation token.
    pub truncated: bool,
    /// Stopped because the cancel flag was set. The text so far is still
    /// returned — the caller decides whether a partial response is worth
    /// parsing.
    pub cancelled: bool,
}

// ── the model ───────────────────────────────────────────────────────────────

/// A loaded GGUF, ready to be sampled.
///
/// Loading is the expensive part (weights off disk and onto the device), so a
/// run loads once and generates many times. The backend is owned here rather
/// than initialized per call: `LlamaBackend::init` is process-global and
/// initializing it twice is an error.
pub struct Teacher {
    backend: LlamaBackend,
    model: LlamaModel,
    path: PathBuf,
}

impl Teacher {
    /// Loads the GGUF named by `cfg.model_file` out of `storage::teacher_dir()`.
    ///
    /// The name is rejected if it contains a path separator or `..`, so the only
    /// thing this can open is a file sitting directly in the teacher directory.
    /// The check is here rather than at the HTTP layer because this is the
    /// function that touches the filesystem.
    pub fn load(cfg: &TeacherConfig) -> Result<Teacher, String> {
        if cfg.model_file.is_empty() {
            return Err("no teacher model named".to_string());
        }
        if cfg.model_file.contains('/')
            || cfg.model_file.contains('\\')
            || cfg.model_file.contains("..")
        {
            return Err(format!(
                "teacher model must be a bare file name inside the teacher dir, got {:?}",
                cfg.model_file
            ));
        }

        let dir = storage::teacher_dir();
        let path = dir.join(&cfg.model_file);
        if !path.is_file() {
            return Err(format!(
                "teacher model {:?} not found in {}",
                cfg.model_file,
                dir.display()
            ));
        }

        let mut backend = LlamaBackend::init().map_err(|e| format!("llama backend init: {e}"))?;
        // llama.cpp logs model metadata and per-eval timings to stderr by
        // default, which over a long run buries the job's own output.
        backend.void_logs();

        let model_params = LlamaModelParams::default().with_n_gpu_layers(cfg.n_gpu_layers);
        let model = LlamaModel::load_from_file(&backend, &path, &model_params)
            .map_err(|e| format!("loading {}: {e}", path.display()))?;

        Ok(Teacher {
            backend,
            model,
            path,
        })
    }

    /// Whether this build can actually offload to the GPU.
    ///
    /// Worth reporting rather than inferring from throughput: a CUDA-less build
    /// runs the same code correctly on the CPU at roughly two orders of
    /// magnitude less speed, which reads as a hang, not a misconfiguration.
    pub fn supports_gpu(&self) -> bool {
        self.backend.supports_gpu_offload()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The window the weights were trained for — the ceiling on what `n_ctx` can
    /// usefully ask for.
    pub fn train_ctx(&self) -> u32 {
        self.model.n_ctx_train()
    }

    /// Renders `system` + `user` through the **model's own** chat template.
    ///
    /// Using the template baked into the GGUF rather than hand-writing one is
    /// not a convenience: an instruct model is only reliable inside the exact
    /// control-token layout it was tuned on, and every family spells that
    /// differently. Hand-rolling it is how a model silently degrades into
    /// completion behaviour — it still answers, just worse, and nothing in the
    /// output says why.
    pub fn chat_prompt(&self, system: Option<&str>, user: &str) -> Result<String, String> {
        let template = self
            .model
            .chat_template(None)
            .map_err(|e| format!("this GGUF carries no chat template: {e}"))?;

        let mut messages = Vec::new();
        if let Some(s) = system.map(str::trim).filter(|s| !s.is_empty()) {
            messages.push(
                LlamaChatMessage::new("system".to_string(), s.to_string())
                    .map_err(|e| format!("system message: {e}"))?,
            );
        }
        messages.push(
            LlamaChatMessage::new("user".to_string(), user.to_string())
                .map_err(|e| format!("user message: {e}"))?,
        );

        // `true` appends the assistant cue, so generation starts inside the reply
        // rather than having to predict the turn header first.
        self.model
            .apply_chat_template(&template, &messages, true)
            .map_err(|e| format!("applying chat template: {e}"))
    }

    /// Opens a reusable generation session.
    ///
    /// The context — and its KV cache, the largest allocation after the weights —
    /// is created once per session and reset between generations rather than
    /// rebuilt per call. It borrows the model, which is why this is a separate
    /// type rather than a field: a struct holding both the model and a context
    /// into it is self-referential, and a plain borrow states the lifetime
    /// without any of the machinery that would otherwise take.
    pub fn session(&self, cfg: &TeacherConfig) -> Result<Session<'_>, String> {
        let n_ctx = NonZeroU32::new(cfg.n_ctx).ok_or("n_ctx must be non-zero")?;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx))
            // A prompt is decoded in one batch, so the batch must be able to hold
            // the whole window; sized smaller, a long prompt fails at decode
            // rather than at configuration.
            .with_n_batch(cfg.n_ctx)
            .with_n_threads(cfg.n_threads)
            .with_n_threads_batch(cfg.n_threads);

        let ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| format!("creating llama context: {e}"))?;

        Ok(Session {
            model: &self.model,
            ctx,
            n_ctx: cfg.n_ctx,
        })
    }
}

// ── generation ──────────────────────────────────────────────────────────────

/// A live context against a loaded model. One at a time, reused across calls.
pub struct Session<'a> {
    model: &'a LlamaModel,
    ctx: LlamaContext<'a>,
    n_ctx: u32,
}

impl Session<'_> {
    /// Generates a completion for an already-templated prompt.
    ///
    /// `on_token` receives each decoded fragment as it arrives, so a caller can
    /// stream progress instead of waiting for the whole response. Fragments, not
    /// tokens: a multi-byte character can straddle two tokens, so bytes are
    /// buffered and released only once they form valid UTF-8.
    ///
    /// Cancellation is checked once per token — the natural step boundary,
    /// matching the convention `llm::train` and `llm::eval::run` use. A
    /// cancelled call returns what it has with `cancelled: true` rather than an
    /// error, because "stopped early" is an outcome, not a failure.
    pub fn generate(
        &mut self,
        prompt: &str,
        opts: &SampleOpts,
        cancel: Option<&AtomicBool>,
        mut on_token: impl FnMut(&str),
    ) -> Result<Generated, String> {
        // Every call starts from an empty cache. Without this the previous prompt
        // stays resident and the model conditions on it — which does not error,
        // it quietly answers the wrong question.
        self.ctx.clear_kv_cache();

        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| format!("tokenizing the prompt: {e}"))?;

        let n_prompt = tokens.len();
        if n_prompt >= self.n_ctx as usize {
            return Err(format!(
                "prompt is {n_prompt} tokens but the context window is {} — \
                 raise n_ctx or shorten the prompt",
                self.n_ctx
            ));
        }
        // What is left of the window after the prompt. Asking for more than this
        // would run past the KV cache mid-generation.
        let room = self.n_ctx as usize - n_prompt;
        let budget = opts.max_tokens.min(room);

        let mut batch = LlamaBatch::new(self.n_ctx as usize, 1);
        let last = n_prompt - 1;
        for (i, token) in tokens.iter().enumerate() {
            // Logits are only needed at the final prompt position — that is the
            // distribution the first generated token comes from. Requesting them
            // everywhere costs a vocab-sized row per position and buys nothing.
            batch
                .add(*token, i as i32, &[0], i == last)
                .map_err(|e| format!("filling the prompt batch: {e}"))?;
        }
        self.ctx
            .decode(&mut batch)
            .map_err(|e| format!("decoding the prompt: {e}"))?;

        let mut sampler = build_sampler(self.model, opts);

        let mut out = String::new();
        // Bytes of a character that spans more than one token, held until they
        // decode. `token_to_bytes` can hand back a fragment that is not valid
        // UTF-8 on its own, and pushing that through `from_utf8_lossy` per token
        // would replace it with U+FFFD instead of joining it to its other half.
        let mut pending: Vec<u8> = Vec::new();
        let mut produced = 0usize;
        let mut pos = n_prompt as i32;
        let mut truncated = false;
        let mut cancelled = false;

        loop {
            if produced >= budget {
                // Either the caller's cap or the end of the window; both read the
                // same way downstream — the text is incomplete.
                truncated = true;
                break;
            }
            if cancel.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                cancelled = true;
                break;
            }

            let token = sampler.sample(&self.ctx, -1);
            if self.model.is_eog_token(token) {
                break;
            }
            sampler.accept(token);

            let bytes = token_bytes(self.model, token)?;
            pending.extend_from_slice(&bytes);
            if let Some(text) = take_utf8(&mut pending) {
                on_token(&text);
                out.push_str(&text);
            }

            produced += 1;

            batch.clear();
            batch
                .add(token, pos, &[0], true)
                .map_err(|e| format!("feeding back a generated token: {e}"))?;
            pos += 1;
            self.ctx
                .decode(&mut batch)
                .map_err(|e| format!("decoding a generated token: {e}"))?;
        }

        // A trailing partial character means the model stopped mid-codepoint.
        // Emitting it lossily is right here — it is the end of the text and there
        // is no further token coming to complete it.
        if !pending.is_empty() {
            let tail = String::from_utf8_lossy(&pending).into_owned();
            on_token(&tail);
            out.push_str(&tail);
        }

        Ok(Generated {
            text: out,
            tokens: produced,
            truncated,
            cancelled,
        })
    }
}

// The raw bytes one token spells.
//
// `token_to_bytes` would do this in one call but is deprecated, and the piece API
// it forwards to reports "your buffer was too small" as an error carrying the
// *negative* of the size it needed — so the retry has to be written out. Starting
// at 32 bytes means effectively every token succeeds first time; the retry exists
// for the rare long one rather than as the common path.
//
// `special` is false: a control token must not come back as its literal text. The
// teacher's own markers are scaffold, and rendering them into the output would put
// `<|im_end|>` inside a conversation we then train on.
fn token_bytes(model: &LlamaModel, token: LlamaToken) -> Result<Vec<u8>, String> {
    match model.token_to_piece_bytes(token, 32, false, None) {
        Ok(bytes) => Ok(bytes),
        Err(TokenToStringError::InsufficientBufferSpace(needed)) => {
            let size = needed.unsigned_abs() as usize;
            model
                .token_to_piece_bytes(token, size, false, None)
                .map_err(|e| format!("decoding a generated token: {e}"))
        }
        Err(e) => Err(format!("decoding a generated token: {e}")),
    }
}

// Splits off the longest valid UTF-8 prefix of `buf`, leaving any incomplete
// trailing character behind for the next token to complete.
//
// Returns `None` rather than an empty string when nothing is ready, so the caller
// can skip the callback entirely on a token that carried half a character.
fn take_utf8(buf: &mut Vec<u8>) -> Option<String> {
    let good = match std::str::from_utf8(buf) {
        Ok(_) => buf.len(),
        // `valid_up_to` is the boundary; whatever follows is either an incomplete
        // tail (wait for more) or genuinely invalid (it will never become valid,
        // and gets flushed lossily at the end of generation).
        Err(e) => e.valid_up_to(),
    };
    if good == 0 {
        return None;
    }
    let rest = buf.split_off(good);
    let text = String::from_utf8(std::mem::replace(buf, rest)).ok()?;
    Some(text)
}

// The sampler chain, in the order llama.cpp applies it: the truncating steps
// narrow the distribution, then `dist` draws from what survives.
//
// `penalties` is deliberately absent and repetition is handled by DRY instead. A
// flat repetition penalty pushes down every token already seen, which on
// structured output means it starts fighting the markup — closing tags and
// delimiters are *supposed* to repeat. DRY penalizes repeated *sequences*, so it
// suppresses a paraphrased duplicate answer while leaving `<|/codeblock|>` alone.
fn build_sampler(model: &LlamaModel, opts: &SampleOpts) -> LlamaSampler {
    let mut chain = Vec::new();

    if opts.dry_multiplier > 0.0 {
        // Argument order is (multiplier, base) — not (base, multiplier). Swapping
        // them compiles fine and silently trades a 0.8 multiplier over a 1.75 base
        // for the reverse, which is a far harsher penalty than intended.
        //
        // `-1` for `penalty_last_n` means "the whole context", which is what we
        // want: the duplicate answers this exists to prevent are separated by a
        // whole reply, so a short window would not see the repetition at all.
        //
        // No sequence breakers. The usual defaults are newline and punctuation,
        // but here the repeated sequences worth catching *span* those — the second
        // answer restating the first paragraph — and a breaker resets the match.
        // The turbofish is required: an empty slice gives the compiler nothing to
        // infer `impl AsRef<[u8]>` from.
        chain.push(LlamaSampler::dry(
            model,
            opts.dry_multiplier,
            opts.dry_base,
            opts.dry_allowed_length,
            -1,
            Vec::<&str>::new(),
        ));
    }
    if opts.top_k > 0 {
        chain.push(LlamaSampler::top_k(opts.top_k));
    }
    if opts.top_p < 1.0 {
        chain.push(LlamaSampler::top_p(opts.top_p, 1));
    }
    if opts.min_p > 0.0 {
        chain.push(LlamaSampler::min_p(opts.min_p, 1));
    }
    chain.push(LlamaSampler::temp(opts.temperature));
    chain.push(LlamaSampler::dist(opts.seed));

    LlamaSampler::chain_simple(chain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_is_held_until_complete() {
        // "é" is two bytes; a token boundary between them must not produce U+FFFD.
        let mut buf = vec![0xC3];
        assert_eq!(take_utf8(&mut buf), None);
        buf.push(0xA9);
        assert_eq!(take_utf8(&mut buf).as_deref(), Some("é"));
        assert!(buf.is_empty());
    }

    #[test]
    fn utf8_releases_the_valid_prefix_only() {
        let mut buf = b"ok".to_vec();
        buf.push(0xE2); // first byte of a three-byte character
        assert_eq!(take_utf8(&mut buf).as_deref(), Some("ok"));
        assert_eq!(buf, vec![0xE2]);
    }

    #[test]
    fn a_model_file_with_a_path_is_refused() {
        let cfg = TeacherConfig {
            model_file: "../../etc/passwd".to_string(),
            ..Default::default()
        };
        // `let Err(…) else` rather than `unwrap_err`: that would need `Teacher` to be
        // `Debug`, and it wraps a `LlamaModel` and a backend handle that are not.
        //
        // Safe to run without a GPU or a model on disk — `load` checks the name
        // before it touches the filesystem or initializes the backend, which is the
        // order this test depends on.
        let Err(err) = Teacher::load(&cfg) else {
            panic!("a model file containing a path must be refused");
        };
        assert!(err.contains("bare file name"), "{err}");
    }
}

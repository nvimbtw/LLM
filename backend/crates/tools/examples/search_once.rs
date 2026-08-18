//! Run one search through the real registry and print what the model would get.
//!
//!     cd backend && cargo run -p tools --example search_once -- "how tall is elbrus"
//!
//! Reads the same environment the server does, so it answers the question the
//! unit tests cannot: whether *this* machine's instance, with its engines and
//! its settings, returns something a model can use — and whether it fits.
//!
//! The size line is the point. A result several times the model's context
//! window is not an error anywhere; it just quietly evicts the conversation it
//! was fetched for. See `SEARCH_BUDGET_CHARS`.

use tools::{Config, Registry};

/// Roughly how many characters a token is worth under this repo's vocabulary —
/// the same figure `scripts/chat_markup.py` tunes against for ASCII text.
const CHARS_PER_TOKEN: usize = 3;

#[tokio::main]
async fn main() {
    let query = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let query = if query.is_empty() { "how tall is mount elbrus".to_string() } else { query };

    let config = Config::from_env();
    let registry = Registry::with_config(&config);

    match registry.origins() {
        [] => {
            eprintln!("No tools configured. Set SEARCH_URL (or SEARCH_API_KEY) and try again.");
            std::process::exit(1);
        }
        origins => println!("{}\n", origins.join(", ")),
    }

    let (body, failure) = registry.answer("web-search", &query).await;
    if let Some(e) = failure {
        eprintln!("failed: {e}\n");
    }

    println!("{body}\n");

    let chars = body.chars().count();
    println!(
        "{chars} chars, ~{} tokens, {} result(s) — budget {} chars",
        chars / CHARS_PER_TOKEN,
        body.matches("<|item|>").count(),
        config.search_budget_chars,
    );
}

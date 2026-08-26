//! What the model would be handed for one command.
//!
//!     cargo run -p tools --example bash_once -- 'echo hi > f.txt; ls'
use tools::{Config, Registry};

#[tokio::main]
async fn main() {
    let command: String = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let registry = Registry::with_config(&Config::from_env(), None);
    let (body, failure) = registry.answer("bash", &command, "example-session").await;
    println!("{body}");
    if let Some(e) = failure {
        eprintln!("failed: {e}");
    }
}

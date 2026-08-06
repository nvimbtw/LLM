// Write the held-out share of `raw/sft/` to `eval/chat/`, without the server.
//
// The queue already does this before every benchmark, so this is not a step anyone
// has to remember — it exists for the same reason `llm`'s `eval` binary does: to
// inspect one stage of the pipeline without standing up the whole backend, and to
// give the markup benchmark its data when that benchmark is being run from the CLI.
//
// Usage:
//   cargo run -p training --bin export_chat_eval
//
// Honours DATA_DIR like everything else that goes through `storage`.

fn main() {
    match training::export_chat_eval_set() {
        Ok(0) => {
            println!("no held-out conversations — is data/raw/sft empty?");
        }
        Ok(n) => {
            let dir = storage::eval_dir().join("chat");
            println!("wrote {n} held-out conversations to {}", dir.display());
        }
        Err(e) => {
            eprintln!("could not export the held-out chat set: {e}");
            std::process::exit(1);
        }
    }
}

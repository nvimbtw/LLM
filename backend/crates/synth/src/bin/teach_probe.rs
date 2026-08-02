// Why is a tier not filling?
//
// A teaching run reports its rejection tally to the live feed and nowhere else,
// so once the job row is written the reason a tier stayed empty is gone. This
// runs a handful of real calls and prints, per answer, which check rejected it —
// the measurement that has to come before changing the prompt, since "the teacher
// ignores the long instruction" and "the long answer is generated and then thrown
// away" call for opposite fixes.
//
//   cargo run -p synth --bin teach_probe -- [calls] [model-uuid]
//
// Writes nothing. Takes the GPU, so do not run it against a busy queue.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use std::path::PathBuf;

use synth::teach::{TeachRequest, probe, replay};
use teacher::{SampleOpts, TeacherConfig};

fn main() {
    let mut args = std::env::args().skip(1);
    let calls: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(3);
    let id_model = args.next().and_then(|a| a.parse().ok());
    let dump: Option<PathBuf> = args.next().map(PathBuf::from);
    let replay_mode = std::env::var("TEACH_PROBE_REPLAY").is_ok();

    let model_file = match std::fs::read_dir(storage::teacher_dir()) {
        // The bare file name, not the path: `TeacherConfig::model_file` is resolved
        // against the teacher dir and refuses anything that looks like a path.
        Ok(entries) => entries
            .flatten()
            .map(|e| e.file_name())
            .find(|n| n.to_string_lossy().ends_with(".gguf"))
            .map(std::path::PathBuf::from),
        Err(e) => {
            eprintln!("could not read the teacher directory: {e}");
            std::process::exit(1);
        }
    };
    let Some(model_file) = model_file else {
        eprintln!("no .gguf in {}", storage::teacher_dir().display());
        std::process::exit(1);
    };
    println!("teacher: {}", model_file.display());
    println!("student: {id_model:?}\n");

    let request = TeachRequest {
        teacher: TeacherConfig {
            model_file: model_file.to_string_lossy().to_string(),
            ..TeacherConfig::default()
        },
        sample: SampleOpts::default(),
        count: calls,
        low_max_tokens: 140,
        med_max_tokens: 700,
        id_model,
        exemplar_file: "gold-general.jsonl".to_string(),
        max_calls: 0,
        seed: Some(7),
    };

    // Unused by `probe`, but it is the same shape the real run takes.
    let _ = Arc::new(AtomicBool::new(false));

    // `replay <dir>` re-parses saved responses instead of generating: no GPU, and
    // an exact A/B when the parser changes.
    let result = match dump.as_deref() {
        Some(d) if replay_mode => replay(&request, d),
        _ => probe(&request, calls, dump.as_deref()),
    };
    match result {
        Ok(lines) => {
            for line in lines {
                print!("{line}");
            }
        }
        Err(e) => {
            eprintln!("probe failed: {e}");
            std::process::exit(1);
        }
    }
}

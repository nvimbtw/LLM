// Run the benchmark suite against a model directory, without the server.
//
// The web UI is the normal way in, but a CLI is useful for the cases the UI can't
// cover: checking a checkpoint that was never registered in the database, running
// the suite under a profiler, or diffing two models from a shell script.
//
// Usage:
//   cargo run -p llm --bin eval -- <model_dir> [eval_dir]
//
// `model_dir` holds `weights` + `merges.bin` (i.e. data/artifacts/models/<uuid>).
// `eval_dir` defaults to `data/eval` and holds `corpus/*.txt` (held-out reference
// prose) and optionally `blimp/*.jsonl`.

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: eval <model_dir> [eval_dir]");
        std::process::exit(2);
    }

    let model_dir = PathBuf::from(&args[1]);
    let eval_dir = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/eval"));

    let config = llm::eval::EvalConfig::new(model_dir, &eval_dir);
    // Nothing can flip a cancel flag in a one-shot CLI run — Ctrl-C kills the
    // process outright — so there is none to pass, and `Ok(None)` is unreachable
    // here rather than a case worth inventing an exit code for.
    let report = match llm::eval::run(&config, Box::new(print_progress), None) {
        Ok(Some(report)) => report,
        Ok(None) => unreachable!("a CLI run has no cancel flag to trip"),
        Err(e) => {
            eprintln!("benchmark failed: {e}");
            std::process::exit(1);
        }
    };

    print_report(&report);
}

// Progress goes to stderr so stdout stays a clean report — `eval … > report.txt`
// should capture results, not a progress log.
fn print_progress(stage: &str, fraction: f64) {
    eprintln!("[{:3.0}%] {}", fraction * 100.0, stage);
}

fn print_report(report: &llm::eval::Report) {
    let m = &report.model;
    println!("\n── model ────────────────────────────────────────────");
    println!(
        "  {}d · {}L · {}h · ctx {} · vocab {} · {} params",
        m.dimensions, m.layers, m.heads, m.context, m.vocab, m.parameters
    );

    println!("\n── held-out text ────────────────────────────────────");
    match &report.corpus {
        Some(c) => {
            println!("  bits/byte        {:.3}", c.bits_per_byte);
            println!("  5-gram floor     {:.3}", c.baseline_bits_per_byte);
            println!("  advantage        {:.1}%", c.advantage * 100.0);
            println!("  perplexity       {:.1} per token", c.perplexity);
            println!("  scored           {} tokens / {} bytes", c.tokens_scored, c.bytes_scored);
        }
        None => println!("  (no reference corpus installed)"),
    }

    println!("\n── grammar (chance = 50%) ───────────────────────────");
    println!(
        "  overall          {:.1}%  ({} pairs)",
        report.grammar.accuracy * 100.0,
        report.grammar.pairs
    );
    for paradigm in &report.grammar.paradigms {
        println!(
            "    {:<32} {:>5.1}%  ({})",
            paradigm.name,
            paradigm.accuracy * 100.0,
            paradigm.pairs
        );
    }

    let g = &report.generation;
    println!("\n── generation ───────────────────────────────────────");
    println!("  real words       {:.1}%", g.real_word_rate * 100.0);
    println!("  distinct-3       {:.1}%", g.distinct_3 * 100.0);
    println!("  repetition       {:.1}%", g.repeat_rate * 100.0);
    // Unknown when no sample attempted markup at all — printing "0.0%" or "NaN%"
    // would both read as a verdict, and it isn't one.
    match g.marker_validity {
        v if v.is_finite() => println!("  markup valid     {:.1}%", v * 100.0),
        _ => println!("  markup valid     —  (no sample emitted markup)"),
    }
    for sample in g.samples.iter().take(3) {
        println!("    {}⟨{}⟩", sample.prompt, sample.completion.trim());
    }

    println!("\n── preference ───────────────────────────────────────");
    match &report.preference {
        Some(p) => {
            println!("  approved         {:.3} logprob/token  ({} replies)", p.approved_logprob, p.approved);
            println!("  rejected         {:.3} logprob/token  ({} replies)", p.rejected_logprob, p.rejected);
            println!("  margin           {:+.3}", p.margin);
            if p.pairs > 0 {
                println!(
                    "  paired accuracy  {:.1}%  ({} judgements, chance = 50%)",
                    p.paired_accuracy * 100.0,
                    p.pairs
                );
            }
        }
        None => println!("  (no thumbs-up/down feedback collected yet)"),
    }

    println!("\n── score ────────────────────────────────────────────");
    println!("  {:.1} / 100   ({:.1}s)", report.score, report.duration_secs);
    println!("  (preference is reported but deliberately not part of the score)");
}

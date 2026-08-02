// Run the benchmark suite against a model directory, without the server.
//
// The web UI is the normal way in, but a CLI is useful for the cases the UI can't
// cover: checking a checkpoint that was never registered in the database, running
// the suite under a profiler, or diffing two models from a shell script.
//
// Usage:
//   cargo run -p llm --bin eval -- [--markup|--assistant] <model_dir> [eval_dir]
//
// `model_dir` holds `weights` + `merges.bin` (i.e. data/artifacts/models/<uuid>).
// `eval_dir` defaults to `data/eval` and holds `corpus/*.txt` (held-out reference
// prose) and optionally `blimp/*.jsonl` and `chat/*.jsonl`.
//
// Three suites, three separate scores, on purpose — they share a 0–100 range and
// nothing else, and a supervised fine-tune is expected to move them in different
// directions. One blended number could not say so.
//
//   (default)     language  — can it model English
//   --markup      markup    — can it drive its own harness
//   --assistant   assistant — are the replies any good (needs `chat/*.jsonl`)

use std::path::PathBuf;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let markup = args.iter().any(|a| a == "--markup");
    let assistant = args.iter().any(|a| a == "--assistant");
    args.retain(|a| a != "--markup" && a != "--assistant");
    if args.is_empty() || (markup && assistant) {
        eprintln!("usage: eval [--markup|--assistant] <model_dir> [eval_dir]");
        std::process::exit(2);
    }

    let model_dir = PathBuf::from(&args[0]);
    let eval_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/eval"));

    let config = llm::eval::EvalConfig::new(model_dir, &eval_dir);
    // Nothing can flip a cancel flag in a one-shot CLI run — Ctrl-C kills the
    // process outright — so there is none to pass, and `Ok(None)` is unreachable
    // here rather than a case worth inventing an exit code for.
    if markup {
        match llm::eval::run_markup(&config, Box::new(print_progress), None) {
            Ok(Some(report)) => print_markup_report(&report),
            Ok(None) => unreachable!("a CLI run has no cancel flag to trip"),
            Err(e) => {
                eprintln!("benchmark failed: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    if assistant {
        match llm::eval::run_assistant(&config, Box::new(print_progress), None) {
            Ok(Some(report)) => print_assistant_report(&report),
            Ok(None) => unreachable!("a CLI run has no cancel flag to trip"),
            Err(e) => {
                eprintln!("benchmark failed: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    match llm::eval::run_language(&config, Box::new(print_progress), None) {
        Ok(Some(report)) => print_report(&report),
        Ok(None) => unreachable!("a CLI run has no cancel flag to trip"),
        Err(e) => {
            eprintln!("benchmark failed: {e}");
            std::process::exit(1);
        }
    }
}

fn print_assistant_report(report: &llm::eval::AssistantReport) {
    println!("\n── model ──");
    println!(
        "  {}d · {}L · {}h · ctx {} · vocab {} · {} params",
        report.model.dimensions,
        report.model.layers,
        report.model.heads,
        report.model.context,
        report.model.vocab,
        report.model.parameters,
    );

    match &report.chat {
        Some(c) => {
            println!("\n── replies ──");
            // Printed with its direction spelled out. The unit runs the opposite way
            // to the score beneath it, which is exactly the confusion the suite was
            // built to end.
            println!(
                "  {:.4} bits/token  (lower is better) ± {:.4}",
                c.bits_per_token, c.bits_per_token_stderr,
            );
            println!("  {} tokens over {} conversations", c.tokens_scored, c.conversations);
            println!("  mean logprob          {:.4}", c.logprob);
        }
        None => println!("\n── replies ──\n  no held-out chat set installed"),
    }

    let b = &report.breakdown;
    println!("\n── score ──");
    println!("  replies               {:.4}", b.replies);
    println!(
        "  ASSISTANT             {:.2} / 100  ± {:.2}   (v{})",
        b.score, b.score_stderr, b.version,
    );
    println!("\n  ran in {:.1}s", report.duration_secs);
}

fn print_markup_report(report: &llm::eval::MarkupReport) {
    let m = &report.markup;
    println!("\n── model ──");
    println!(
        "  {}d · {}L · {}h · ctx {} · vocab {} · {} params",
        report.model.dimensions,
        report.model.layers,
        report.model.heads,
        report.model.context,
        report.model.vocab,
        report.model.parameters,
    );

    println!("\n── marker pairs ──");
    println!("  overall        {}  ({} pairs, chance 50%)", pct(m.pairs.accuracy), m.pairs.pairs);
    for paradigm in &m.pairs.paradigms {
        println!("    {:<32} {}  ({})", paradigm.name, pct(paradigm.accuracy), paradigm.pairs);
    }

    match &m.positions {
        Some(p) => {
            println!("\n── marker positions ──");
            println!("  top-1 at markers      {}  ({} positions)", pct(p.marker_top1), p.marker_positions);
            println!("  restricted top-1      {}", pct(p.marker_restricted_top1));
            println!("  mean rank             {:.2}", p.marker_mean_rank);
            println!("  false-marker rate     {}  ({} prose positions)", pct(p.false_marker_rate), p.prose_positions);
            println!("  conversations         {}", p.conversations);
        }
        None => println!("\n── marker positions ──\n  no held-out chat set installed"),
    }

    match &m.structure {
        Some(s) => {
            println!("\n── generated structure ──");
            println!("  closed the turn       {}  ({} replies)", pct(s.close_rate), s.replies);
            println!("  balanced markup       {}", pct(s.balance_rate));
            println!(
                "  marker density        {:.2} per 100 tokens (reference {:.2})",
                s.marker_density, s.reference_marker_density,
            );
            for sample in s.samples.iter().take(3) {
                println!("\n    > {}", truncate(&sample.prompt, 100));
                println!("      {}", truncate(&sample.completion, 200));
            }
        }
        None => println!("\n── generated structure ──\n  no held-out chat set installed"),
    }

    match &report.chat {
        // Reported, never scored — a base model has never seen the chat scaffold.
        Some(c) => println!(
            "\n── reply bits (not scored) ──\n  {:.4} bits/token over {} tokens in {} conversations",
            c.bits_per_token, c.tokens_scored, c.conversations,
        ),
        None => println!("\n── reply bits (not scored) ──\n  no held-out chat set installed"),
    }

    let b = &report.breakdown;
    println!("\n── score ──");
    println!("  pairs      {:.3}", b.pairs);
    println!("  positions  {:.3}", b.positions);
    println!("  structure  {:.3}", b.structure);
    println!("  restraint  {:.3}", b.restraint);
    println!(
        "\n  {:.1} ± {:.1} / 100   (markup v{}, {:.1}s)",
        b.score, b.score_stderr, b.version, report.duration_secs,
    );
}

fn pct(value: f64) -> String {
    if value.is_finite() {
        format!("{:.1}%", value * 100.0)
    } else {
        "—".to_string()
    }
}

fn truncate(text: &str, max: usize) -> String {
    let cleaned = text.replace('\n', " ");
    match cleaned.char_indices().nth(max) {
        Some((at, _)) => format!("{}…", &cleaned[..at]),
        None => cleaned,
    }
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

// The post-training cycle: the layer above the queue.
//
// A cycle repeats one round until it stops improving:
//
//     generate + score  → autolabel job   (fills the catalogue against the frozen bar)
//     self-train        → train job, RAFT (moves the model toward what cleared it)
//     preference        → train job, KTO  (pushes down what didn't)
//     benchmark         → benchmark job   (scores the result)
//     decide            → no job          (accept, or halt)
//
// **It never touches the GPU.** That is not a style choice: a cycle waits for the
// jobs it spawns, so if it held the device itself its children would queue behind
// it forever. It enqueues, then reacts to completions — which also means a restart
// resumes it, since every piece of its state is a row.
//
// **It is edge-triggered with a periodic backstop.** The driver wakes on the queue's
// change feed and advances whatever is ready; a slow tick covers a missed
// notification. Every transition is idempotent for the same reason — an event may
// arrive twice, and `open_round`'s ON CONFLICT plus the stage checks below mean the
// second one does nothing.
//
// **A round trains on its own generations and no others.** `raw/preference/` is
// append-only and outlives every model and cycle in it, so "the catalogue" means
// every round anything has ever run. Training on all of it makes round N a pass over
// rounds 1..N — the oldest and worst generations collecting the most cumulative
// gradient, which is a ratchet pointing down. Every generated line is therefore
// stamped with the round that wrote it, and both training stages are scoped to it:
// `round_key` names the round, `catalogue_counts` gates on that slice, and the same
// pair rides to the worker on `TrainConfig`. The gate and the build must be scoped
// alike or a gate passes on a catalogue the build cannot produce. Human ratings are
// unscoped — they are the seed and the audit, not a round's output.
//
// **The round is `(id_cycle, round)`, and it used to be the generating model.** That
// was wrong and silently did nothing: every re-run of a cycle starts from the same
// seed and generates with it, so every run stamped the same `id_model` and the scope
// never separated them. Four runs of one cycle put 1160 lines under one id; the
// catalogue grew 320 → 604 → 900 and the round score fell 58.39 → 56.85 → 54.41 in
// step with it. `cycle_rounds.id_model_in` remains as provenance and is no longer
// what anything is scoped by.
//
// **Ordering matters and is not arbitrary.** RAFT runs before KTO because after
// RAFT the approved replies already sit at high probability, so KTO's gradient on
// them saturates (see `llm::kto::step`) and it spends its budget almost entirely on
// suppressing the rejected half. The reverse order would have them fighting over
// the same examples.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tokio::sync::Notify;
use uuid::Uuid;

use db;
use db::cycle::{CycleStage, CycleStatus};
use db::job::{JobKind, JobStatus};
use labeling::gather;
use crate::queue::Queue;
use training::{TrainConfig, TrainMode};

// Backstop behind the change feed. Everything that finishes a job also notifies, so
// this only matters when a notification is missed.
const IDLE_POLL: Duration = Duration::from_secs(15);

// Knobs a cycle runs under, frozen into its row at creation so a run records the
// rules it followed. Defaults match what the composer sends by hand.
#[derive(Clone, Deserialize, Serialize)]
pub struct CycleConfig {
    #[serde(default = "default_prompts")]
    pub max_prompts: i64,
    #[serde(default = "default_candidates")]
    pub candidates: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    // How long a generated candidate reply may run, in tokens.
    //
    // **The single most consequential number in the cycle**, and for a long time it was
    // not a knob at all — `labeling::autolabel` hard-coded 96, and no cycle setting
    // could reach it. A round therefore trained on replies averaging 88 characters
    // while the assistant benchmark scored replies averaging 1400, so every round
    // moved the model further from what it was measured on. See
    // `autolabel::DEFAULT_MAX_TOKENS` for the measurements.
    //
    // Raising it makes the generation stage several times slower — it is the cycle's
    // expensive stage — which is the trade being made deliberately.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_raft_lr")]
    pub raft_lr: f32,
    #[serde(default = "default_kto_lr")]
    pub kto_lr: f32,
    #[serde(default = "default_beta")]
    pub beta: f64,
    // How much the benchmark score must rise for a round to count as an
    // improvement. Not zero: benchmark noise alone will produce a positive delta
    // about half the time, and a loop that accepts noise runs until it drifts.
    //
    // On the composite's own 0–100 scale. It was 0.005 — a hundredth of a point,
    // which is not a noise floor but a rounding error, and it let rounds turn on
    // movement far smaller than the ±1-point swings the surface metrics show
    // between adjacent checkpoints.
    #[serde(default = "default_min_gain")]
    pub min_gain: f64,
    // Run every round to the limit instead of halting on the first that does not
    // improve. **Nothing about the measurement changes** — `judge` runs, the verdict
    // is recorded, a failed round is written as `explored` rather than `accepted`, and
    // `best_score` still only moves on a genuine improvement. The only difference is
    // that a failing round no longer stops the cycle.
    //
    // This is for answering "where does this go over 5-10 rounds" — whether the loop
    // recovers, plateaus or degrades monotonically — which a halt at round 1 cannot
    // show. It is an experiment mode, not a looser bar: the numbers in the round table
    // are the same numbers the gate would have halted on.
    #[serde(default)]
    pub explore: bool,
    // Run the round as generate → self-train → benchmark, with no preference stage.
    //
    // **An experiment knob, and the measurement it exists for.** Benchmarking the
    // intermediate model of one round put the two training stages at −3.82 (RAFT) and
    // −5.74 (KTO) points of assistant score, so KTO is the larger single loss and the
    // only stage in the pipeline with no prose replay anchoring it. RAFT's −3.82 is
    // 1.4σ — inside noise — so whether the self-train half is salvageable on its own
    // is an open question that one round with this set answers.
    //
    // With this on, the round's output *is* the RAFT model, so the benchmark stage
    // scores it directly and no extra plumbing is needed to see the split.
    #[serde(default)]
    pub skip_preference: bool,
}

fn default_prompts() -> i64 {
    80
}
fn default_candidates() -> usize {
    6
}
fn default_temperature() -> f32 {
    1.0
}
fn default_max_tokens() -> usize {
    512
}
fn default_raft_lr() -> f32 {
    0.00002
}
fn default_kto_lr() -> f32 {
    0.000005
}
fn default_beta() -> f64 {
    0.1
}
fn default_min_gain() -> f64 {
    0.5
}

impl Default for CycleConfig {
    fn default() -> Self {
        CycleConfig {
            max_prompts: default_prompts(),
            candidates: default_candidates(),
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
            raft_lr: default_raft_lr(),
            kto_lr: default_kto_lr(),
            beta: default_beta(),
            min_gain: default_min_gain(),
            explore: false,
            skip_preference: false,
        }
    }
}

pub struct Cycles {
    pool: PgPool,
    queue: Arc<Queue>,
    wake: Notify,
}

impl Cycles {
    pub fn new(pool: PgPool, queue: Arc<Queue>) -> Cycles {
        Cycles {
            pool,
            queue,
            wake: Notify::new(),
        }
    }

    pub fn wake(&self) {
        self.wake.notify_one();
    }

    // The driver loop. Owns the pool, does no compute, and is the only thing that
    // writes a cycle's stage.
    pub fn spawn(self: &Arc<Self>) {
        let driver = self.clone();
        tokio::spawn(async move {
            let mut events = driver.queue.subscribe();
            loop {
                driver.tick().await;
                tokio::select! {
                    _ = driver.wake.notified() => {}
                    // Any board change may mean a child job finished. Cheap to
                    // re-examine: one query per active cycle, and there is rarely
                    // more than one.
                    _ = events.recv() => {}
                    _ = tokio::time::sleep(IDLE_POLL) => {}
                }
            }
        });
    }

    async fn tick(&self) {
        let cycles = match db::cycle::list_active(&self.pool).await {
            Ok(cycles) => cycles,
            Err(e) => {
                eprintln!("[cycle] could not list active cycles: {e}");
                return;
            }
        };

        for cycle in cycles {
            if cycle.status != CycleStatus::Running {
                continue;
            }
            if let Err(e) = self.advance(&cycle).await {
                eprintln!("[cycle] {} failed to advance: {e}", cycle.id_cycle);
                self.halt(cycle.id_cycle, &e).await;
            }
        }
    }

    // One step for one cycle. Either the current stage is still waiting on its job
    // (do nothing), or it has finished and the next stage begins.
    async fn advance(&self, cycle: &db::cycle::Cycle) -> Result<(), String> {
        if let Some(id_job) = cycle.id_job_current {
            let job = db::job::find_by_id(&self.pool, id_job)
                .await
                .map_err(|e| format!("could not read job {id_job}: {e}"))?;
            let Some(job) = job else {
                // The row was deleted out from under the cycle. Nothing left to
                // wait for and no result to use.
                return Err(format!("job {id_job} vanished"));
            };

            match job.status {
                JobStatus::Done => {}
                JobStatus::Failed => {
                    return Err(format!(
                        "the {} job failed: {}",
                        stage_name(cycle.stage),
                        job.error.as_deref().unwrap_or("no reason recorded")
                    ));
                }
                JobStatus::Cancelled => {
                    // A person stopped it. That is an instruction, not a fault:
                    // pause rather than halt, so resuming picks up here.
                    self.pause(cycle.id_cycle).await;
                    return Ok(());
                }
                // Still queued or running — nothing to do until it lands.
                _ => return Ok(()),
            }

            return self.on_stage_done(cycle, &job).await;
        }

        // No job outstanding: start the current stage.
        self.start_stage(cycle).await
    }

    // Kicks off whatever the cycle's current stage needs.
    async fn start_stage(&self, cycle: &db::cycle::Cycle) -> Result<(), String> {
        let config: CycleConfig = serde_json::from_value(cycle.config.clone())
            .map_err(|e| format!("stored cycle config is unreadable: {e}"))?;
        let model = cycle
            .id_model_current
            .ok_or_else(|| "the cycle has no current model".to_string())?;

        match cycle.stage {
            CycleStage::Generating => {
                // Scoped to this round's own model, which has generated nothing yet
                // — so this pair is the round's starting point (normally zero from
                // the catalogue, plus whatever human ratings exist), and
                // `record_catalogue` at the end of the stage measures the yield
                // against it. That is what the before/after pair was always for; it
                // is only now a delta rather than two running totals.
                let (approved, rejected) =
                    self.catalogue_counts(Some(Self::round_key(cycle))).await?;
                db::cycle::open_round(
                    &self.pool,
                    cycle.id_cycle,
                    cycle.round,
                    model,
                    approved,
                    rejected,
                )
                .await
                .map_err(|e| format!("could not open round: {e}"))?;

                let request = labeling::autolabel::AutolabelRequest {
                    id_model: model,
                    // Stamped onto every line this pass writes, so the training
                    // stages can select this round's output and nothing else.
                    id_cycle: Some(cycle.id_cycle),
                    round: Some(cycle.round),
                    candidates: config.candidates,
                    max_prompts: config.max_prompts,
                    temperature: config.temperature,
                    max_tokens: config.max_tokens,
                    // Never from inside a cycle. The frozen bar is what makes
                    // "fraction clearing it" comparable across rounds; refitting
                    // mid-run would move the yardstick under the measurement.
                    recalibrate: false,
                };
                let id_job = self
                    .enqueue(
                        cycle,
                        JobKind::Autolabel,
                        &format!("{} r{} catalogue", cycle.label, cycle.round),
                        serde_json::to_value(&request).map_err(|e| e.to_string())?,
                        "generating",
                    )
                    .await?;
                self.set_stage(cycle.id_cycle, CycleStage::Generating, Some(id_job))
                    .await;
            }

            CycleStage::Selftrain => {
                // The round's generating model is still `id_model_current` here —
                // it is this stage's output that replaces it. Read it back through
                // the round row anyway, so both training stages resolve the slice
                // the same way and there is one answer to "what is this round's
                // catalogue" rather than two that happen to agree.
                let catalogue = Some(Self::round_key(cycle));
                let (approved, _) = self.catalogue_counts(catalogue).await?;
                gather::check_raft(approved)?;
                let cfg = self
                    .train_config(
                        TrainMode::Raft,
                        model,
                        catalogue,
                        &format!("{} r{} self", cycle.label, cycle.round),
                        config.raft_lr,
                        gather::max_epochs_for(approved),
                        config.beta,
                    )
                    .await?;
                let id_job = self
                    .enqueue(
                        cycle,
                        JobKind::Train,
                        &cfg.model_name.clone(),
                        serde_json::to_value(&cfg).map_err(|e| e.to_string())?,
                        "selftrain",
                    )
                    .await?;
                self.set_stage(cycle.id_cycle, CycleStage::Selftrain, Some(id_job))
                    .await;
            }

            CycleStage::Preference => {
                // Skipped outright, before the catalogue is even counted — there is
                // nothing to gate on when the stage is not going to run. The round's
                // model is then whatever self-train produced, which is exactly the
                // model this flag exists to get a clean score for.
                if config.skip_preference {
                    self.set_stage(cycle.id_cycle, CycleStage::Benchmarking, None)
                        .await;
                    return Ok(());
                }

                let catalogue = Some(Self::round_key(cycle));
                let (approved, rejected) = self.catalogue_counts(catalogue).await?;
                // A catalogue too small or too one-sided for KTO is not a failure —
                // it is the expected end state, since a model that clears the bar
                // more often produces fewer rejections every round. Skip the stage
                // and carry on to the benchmark.
                if gather::check_catalogue(approved, rejected).is_err() {
                    self.set_stage(cycle.id_cycle, CycleStage::Benchmarking, None)
                        .await;
                    return Ok(());
                }
                let cfg = self
                    .train_config(
                        TrainMode::Kto,
                        model,
                        catalogue,
                        &format!("{} r{} pref", cycle.label, cycle.round),
                        config.kto_lr,
                        gather::max_epochs_for(approved + rejected),
                        config.beta,
                    )
                    .await?;
                let id_job = self
                    .enqueue(
                        cycle,
                        JobKind::Train,
                        &cfg.model_name.clone(),
                        serde_json::to_value(&cfg).map_err(|e| e.to_string())?,
                        "preference",
                    )
                    .await?;
                self.set_stage(cycle.id_cycle, CycleStage::Preference, Some(id_job))
                    .await;
            }

            CycleStage::Benchmarking => {
                // All three suites, one job each, and the cycle waits on the last.
                //
                // **Waiting on the last is sound because there is one GPU and one
                // runner.** Jobs are claimed in `queue_rank` order and these three go
                // in back to back, so the third finishing means the first two have. It
                // is not relied on blindly: `decide` re-reads every suite's run from
                // `benchmark_runs` and fails the round outright if one is missing, so a
                // hand-reordered queue produces a legible failure rather than a round
                // accepted on partial evidence.
                let mut last = None;
                for (suite, _, name) in CYCLE_SUITES {
                    last = Some(
                        self.enqueue(
                            cycle,
                            JobKind::Benchmark,
                            &format!("{} r{} {name}", cycle.label, cycle.round),
                            serde_json::json!({
                                "id_model": model,
                                "suite": suite.as_str(),
                            }),
                            "benchmarking",
                        )
                        .await?,
                    );
                }
                self.set_stage(cycle.id_cycle, CycleStage::Benchmarking, last)
                    .await;
            }

            // Reached only if a restart landed here; the decision needs the
            // benchmark job's result, so re-run that stage rather than guess.
            CycleStage::Deciding => {
                self.set_stage(cycle.id_cycle, CycleStage::Benchmarking, None)
                    .await;
            }
        }
        Ok(())
    }

    // A stage's job finished successfully: record what it produced and move on.
    async fn on_stage_done(
        &self,
        cycle: &db::cycle::Cycle,
        job: &db::job::Job,
    ) -> Result<(), String> {
        match cycle.stage {
            CycleStage::Generating => {
                let catalogue = Some(Self::round_key(cycle));
                let (approved, rejected) = self.catalogue_counts(catalogue).await?;
                db::cycle::record_catalogue(
                    &self.pool,
                    cycle.id_cycle,
                    cycle.round,
                    approved,
                    rejected,
                )
                .await
                .map_err(|e| format!("could not record the catalogue: {e}"))?;
                self.set_stage(cycle.id_cycle, CycleStage::Selftrain, None)
                    .await;
            }

            // Both trainers hand their new model forward through `ref_id`, so the
            // next stage tunes (or scores) what the last one produced rather than
            // what the round started from.
            CycleStage::Selftrain => {
                let produced = job
                    .ref_id
                    .ok_or_else(|| "the self-train job registered no model".to_string())?;
                self.set_current_model(cycle.id_cycle, produced).await?;
                self.set_stage(cycle.id_cycle, CycleStage::Preference, None)
                    .await;
            }

            CycleStage::Preference => {
                let produced = job
                    .ref_id
                    .ok_or_else(|| "the preference job registered no model".to_string())?;
                self.set_current_model(cycle.id_cycle, produced).await?;
                self.set_stage(cycle.id_cycle, CycleStage::Benchmarking, None)
                    .await;
            }

            // The scores are not taken from this job: three suites ran, and `decide`
            // reads each one's stored run by (model, suite, version) — which is also
            // the only way to fetch the *bar* model's numbers, since those were
            // recorded rounds ago.
            CycleStage::Benchmarking => {
                self.decide(cycle).await?;
            }

            CycleStage::Deciding => {}
        }
        Ok(())
    }

    // Accept the round or stop. The only place a cycle decides anything; the policy
    // itself is `judge`, below.
    async fn decide(&self, cycle: &db::cycle::Cycle) -> Result<(), String> {
        let config: CycleConfig = serde_json::from_value(cycle.config.clone())
            .map_err(|e| format!("stored cycle config is unreadable: {e}"))?;
        let produced = cycle
            .id_model_current
            .ok_or_else(|| "the round produced no model".to_string())?;

        // Which model this round has to beat. After round 1 that is the best round so
        // far; for round 1 it is the model the cycle started from, looked up rather
        // than re-benchmarked (the seed is a saved model, so its numbers are already
        // on record). Treating round 1 as "nothing to beat" is what let all three
        // recorded cycles accept a first round well below their own seed — one of them
        // 6.6 points below, with grammar falling from 0.598 to 0.492, i.e. below
        // chance.
        let (bar_model, bar_label) = match cycle.id_model_best {
            Some(best) => (Some(best), "the best round's"),
            None => (cycle.id_model_start, "the starting model's"),
        };

        // Every suite at exactly its own scoring version — a run recorded under an
        // older version is not a worse number, it is a different scale, which is what
        // `latest_evidence_for_model`'s filter exists to catch.
        let mut comparisons = Vec::with_capacity(CYCLE_SUITES.len());
        for (suite, version, name) in CYCLE_SUITES {
            let round = db::benchmark::latest_evidence_for_model(
                &self.pool, produced, suite, version,
            )
            .await
            .map_err(|e| format!("could not read this round's {name} benchmark: {e}"))?;

            // The round's own runs all have to be there. One missing means a suite
            // failed, or the queue was reordered so the cycle stopped waiting too
            // early — either way the round is unjudgeable, and continuing would be
            // running blind for hours.
            let Some(round) = round else {
                let note = format!(
                    "the {name} benchmark produced no score at version {version}, so \
                     this round cannot be judged"
                );
                self.close_round(cycle, Some(produced), None, "failed", &note)
                    .await;
                return Err(note);
            };

            // A missing *bar* run is ordinary: the seed may predate the suite. That
            // suite drops out of the comparison rather than failing the round.
            let bar = match bar_model {
                Some(id) => db::benchmark::latest_evidence_for_model(
                    &self.pool, id, suite, version,
                )
                .await
                .map_err(|e| format!("could not read {bar_label} {name} benchmark: {e}"))?,
                None => None,
            };

            comparisons.push(SuiteComparison { name, round, bar });
        }

        // The number the round is filed under. Deliberately the assistant suite's, not
        // a blend: it is the one the cycle is trying to move, and a mean of three
        // would be a composite by the back door — the exact thing the vector test
        // below refuses to decide on. The note carries all three.
        let score = comparisons
            .iter()
            .find(|c| c.name == ASSISTANT_SUITE)
            .map(|c| c.round.score)
            .unwrap_or(f64::NAN);

        let judgement = judge(&comparisons, bar_label, config.min_gain);

        if !judgement.accept && !config.explore {
            self.close_round(
                cycle,
                Some(produced),
                Some(score),
                "no_improvement",
                &judgement.note,
            )
            .await;
            self.halt(cycle.id_cycle, &judgement.note).await;
            return Ok(());
        }

        // In explore mode a failed round is recorded as failed and then carried on
        // from anyway. The verdict is the honest one — `explored` is a distinct
        // outcome from `accepted`, and the note is the same sentence the halt would
        // have carried — so the round table reads as the trajectory it is rather than
        // as a run of successes.
        let verdict = if judgement.accept { "accepted" } else { "explored" };
        self.close_round(cycle, Some(produced), Some(score), verdict, &judgement.note)
            .await;

        // **`best` still only moves on a real improvement**, explore mode included.
        // The chain continues from the round's output either way (that is what makes
        // it a trajectory), but the bar has to stay at the best model actually seen or
        // each round is measured against the last one and cumulative drift reads as a
        // series of small, innocent deltas. Keeping it pinned means round 7's note
        // says how far it is from the best, which is the question being asked.
        if judgement.accept {
            if let Err(e) = db::cycle::set_best(&self.pool, cycle.id_cycle, produced, score).await
            {
                eprintln!("[cycle] could not record the best model: {e}");
            }
        }

        if cycle.round >= cycle.max_rounds {
            if let Err(e) = db::cycle::set_status(
                &self.pool,
                cycle.id_cycle,
                CycleStatus::Done,
                Some("reached the round limit"),
            )
            .await
            {
                eprintln!("[cycle] could not close out cycle: {e}");
            }
            return Ok(());
        }

        db::cycle::advance_round(&self.pool, cycle.id_cycle, produced)
            .await
            .map_err(|e| format!("could not start the next round: {e}"))?;
        self.queue.notify_changed();
        Ok(())
    }

    // ── helpers ─────────────────────────────────────────────────────────────

    // Builds a train config the way `prepare_train` would, including the storage id
    // (reuse the model of the same name, else a fresh one) and the epoch cap. The
    // driver enqueues rows directly rather than posting to its own HTTP API, so the
    // few lines of that route's preparation are repeated here on purpose — the
    // alternative is a server calling itself over the network to queue its own work.
    async fn train_config(
        &self,
        mode: TrainMode,
        base: Uuid,
        catalogue: Option<(Uuid, i32)>,
        name: &str,
        lr: f32,
        epochs: i64,
        beta: f64,
    ) -> Result<TrainConfig, String> {
        let storage_id = db::model::find_id_by_name(&self.pool, name)
            .await
            .map_err(|e| format!("could not resolve the model id: {e}"))?
            .unwrap_or_else(Uuid::new_v4);

        // Architecture fields are placeholders: both modes load the base model's
        // weights and `apply_base_architecture` overwrites these with the real ones
        // before the run is registered.
        Ok(TrainConfig {
            model_name: name.to_string(),
            dimensions: 64,
            ctx_window: 64,
            heads: 4,
            layers: 2,
            epochs,
            lr,
            ff_multiplier: 4,
            max_vocab: 1024,
            mode,
            base_model_id: Some(base),
            beta,
            catalogue_cycle: catalogue.map(|(id, _)| id),
            catalogue_round: catalogue.map(|(_, r)| r),
            datasets: Vec::new(),
            curriculum: Vec::new(),
            shuffle: false,
            // Inert for the two modes this builds: RAFT and KTO train on the model's
            // own generations, which carry no thinking block to strip, so their
            // dataset builders render `Reasoning::Keep` regardless of this field.
            strip_thinking: false,
            seed: 0,
            storage_id,
        })
    }

    async fn enqueue(
        &self,
        cycle: &db::cycle::Cycle,
        kind: JobKind,
        label: &str,
        config: serde_json::Value,
        role: &str,
    ) -> Result<Uuid, String> {
        // Background band: a cycle runs for hours, and every chat turn sent while it
        // does would otherwise wait behind a multi-minute generation pass.
        let job = db::job::create_in_band(
            &self.pool,
            kind,
            label,
            &config,
            None,
            true,
            Some(cycle.id_cycle),
            Some(role),
        )
        .await
        .map_err(|e| format!("could not queue the {role} job: {e}"))?;

        self.queue.wake();
        self.queue.notify_changed();
        Ok(job.id_job)
    }

    // Which round's generations this round's training stages may read.
    //
    // Just the cycle's own identity — no database round-trip, because a round is
    // exactly `(id_cycle, round)` and the cycle is holding both. It used to be the
    // round's *generating model*, read back from `cycle_rounds.id_model_in`, and that
    // did not separate anything: every re-run of a cycle starts from the same seed
    // and generates with it, so every run stamped the same id and each run trained on
    // the union of all the earlier ones. `id_model_in` stays as provenance; it is no
    // longer what anything is scoped by.
    fn round_key(cycle: &db::cycle::Cycle) -> (Uuid, i32) {
        (cycle.id_cycle, cycle.round)
    }

    // The size of the catalogue a training run would actually be handed.
    //
    // The human ratings are counted whole whatever `only` says, matching
    // `training::read_catalogue`: they carry no round stamp, they are the seed and
    // the audit rather than a round's output, and there are a few dozen of them
    // against a few hundred generated lines. These two must agree or a gate passes
    // on a catalogue the build then cannot produce.
    async fn catalogue_counts(&self, only: Option<(Uuid, i32)>) -> Result<(i64, i64), String> {
        let counts = db::signal::counts(&self.pool)
            .await
            .map_err(|e| format!("could not count signals: {e}"))?;
        let (auto_up, auto_down) = labeling::autolabel::catalogue_counts(only);
        Ok((
            counts.desirable + auto_up,
            counts.undesirable + auto_down,
        ))
    }

    async fn set_current_model(&self, id_cycle: Uuid, id_model: Uuid) -> Result<(), String> {
        sqlx::query!(
            "UPDATE cycles SET id_model_current = $2, updated_at = now() WHERE id_cycle = $1",
            id_cycle,
            id_model,
        )
        .execute(&self.pool)
        .await
        .map_err(|e| format!("could not record the produced model: {e}"))?;
        Ok(())
    }

    async fn set_stage(&self, id_cycle: Uuid, stage: CycleStage, id_job: Option<Uuid>) {
        if let Err(e) = db::cycle::set_stage(&self.pool, id_cycle, stage, id_job).await {
            eprintln!("[cycle] could not set stage: {e}");
        }
        self.queue.notify_changed();
    }

    async fn close_round(
        &self,
        cycle: &db::cycle::Cycle,
        id_model_out: Option<Uuid>,
        score: Option<f64>,
        verdict: &str,
        note: &str,
    ) {
        if let Err(e) = db::cycle::close_round(
            &self.pool,
            cycle.id_cycle,
            cycle.round,
            id_model_out,
            score,
            verdict,
            note,
        )
        .await
        {
            eprintln!("[cycle] could not close round: {e}");
        }
    }

    async fn halt(&self, id_cycle: Uuid, reason: &str) {
        if let Err(e) =
            db::cycle::set_status(&self.pool, id_cycle, CycleStatus::Halted, Some(reason)).await
        {
            eprintln!("[cycle] could not halt cycle: {e}");
        }
        self.queue.notify_changed();
    }

    async fn pause(&self, id_cycle: Uuid) {
        if let Err(e) = db::cycle::set_status(
            &self.pool,
            id_cycle,
            CycleStatus::Paused,
            Some("a job was cancelled"),
        )
        .await
        {
            eprintln!("[cycle] could not pause cycle: {e}");
        }
        self.queue.notify_changed();
    }
}

// The name the assistant suite goes by in a note, and the suite whose score a round
// is filed under.
const ASSISTANT_SUITE: &str = "replies";

/// The three suites every round is benchmarked on, each paired with the scoring
/// version it is comparable at. Enqueued in this order; the cycle waits on the last.
///
/// **Three scores, never blended.** A weighted composite of them would let a large
/// gain on one pay for a collapse on another and still report progress — and with
/// three instruments there are more places for that to hide, not fewer. `judge` runs
/// a vector test over them instead, which is what "a balanced high score on all
/// three" actually means when written as a rule.
const CYCLE_SUITES: [(db::benchmark::BenchmarkSuite, i32, &str); 3] = [
    (
        db::benchmark::BenchmarkSuite::Language,
        llm::eval::SCORE_VERSION as i32,
        "language",
    ),
    (
        db::benchmark::BenchmarkSuite::Markup,
        llm::eval::markup::MARKUP_SCORE_VERSION as i32,
        "markup",
    ),
    (
        db::benchmark::BenchmarkSuite::Assistant,
        llm::eval::ASSISTANT_SCORE_VERSION as i32,
        ASSISTANT_SUITE,
    ),
];

// One suite's round-versus-bar pair. `bar` is `None` when that suite has no
// comparable run on record for the model being beaten — which drops it out of the
// test rather than counting as either outcome.
struct SuiteComparison {
    name: &'static str,
    round: db::benchmark::RoundEvidence,
    bar: Option<db::benchmark::RoundEvidence>,
}

// The outcome of the acceptance policy, and the sentence written to the round.
struct Judgement {
    accept: bool,
    note: String,
}

/// Whether a round may be accepted. The whole acceptance policy, as a pure function
/// so it can be tested without a database, a GPU or a six-hour cycle.
///
/// **The rule, over all three suites: nothing regressed, and at least one improved.**
///
/// Not a weighted sum of the three. A sum is precisely the instrument that lets a
/// large gain on one suite pay for a collapse on another and still report progress,
/// and with three numbers there are more places for that to hide rather than fewer.
/// "A balanced high score on all three" is a constraint, not an average, so the gate
/// is written as a constraint: every suite must hold its ground, and the round has to
/// buy something somewhere.
///
/// **Every suite is judged against its own error bar**, the standard error of the
/// *difference* — `sqrt(a² + b²)` over the two runs — so the cycle can never turn on a
/// movement its own benchmark cannot resolve. The old flat `min_gain` of 0.5 sat
/// *below* that floor (the language composite reports ±0.9 to ±1.0 per run, i.e.
/// ±1.37 on a difference), which is how a 0.41σ drop, 58.94 → 58.39, came to halt a
/// cycle. `min_gain` survives only as a *floor* on each margin: it can make the test
/// blunter, never finer than the instrument.
///
/// The same margin decides both directions, so a movement too small to count as a
/// regression is also too small to count as an improvement. That symmetry is what
/// stops the gate from ratcheting on noise.
///
/// **A suite with no comparable run on the bar model drops out** rather than counting
/// as a pass or a fail — the seed may simply predate that suite. If every suite drops
/// out the round is accepted and recorded as *unmeasured*, which is the honest label:
/// it is not an improvement, it is an absence of evidence.
///
/// **What this replaces.** A gate that read the language composite alone, with no term
/// anywhere in it measuring reply quality — so chat tuning was accepted and rejected
/// entirely on how well the model predicted held-out books. The 58.94 → 58.39 halt was
/// `words` **up** 0.094 and grammar flat, outvoted by prose bits per byte.
fn judge(comparisons: &[SuiteComparison], bar_label: &str, min_gain: f64) -> Judgement {
    let mut parts: Vec<String> = Vec::with_capacity(comparisons.len());
    let mut regressed: Vec<&str> = Vec::new();
    let mut improved: Vec<&str> = Vec::new();
    let mut comparable = 0usize;

    for c in comparisons {
        let Some(bar) = &c.bar else {
            parts.push(format!(
                "{} {:.4} (no comparable {bar_label} run)",
                c.name, c.round.score
            ));
            continue;
        };
        comparable += 1;

        // `f64::max` returns the operand that is not NaN, so a report with no error
        // bar falls back to `min_gain` rather than to zero tolerance — which would
        // make an absent measurement look like a perfectly certain one.
        let margin = min_gain.max(stderr_of_difference(c.round.score_stderr, bar.score_stderr));
        let delta = c.round.score - bar.score;

        let verdict = if delta > margin {
            improved.push(c.name);
            "better"
        } else if delta < -margin {
            regressed.push(c.name);
            "worse"
        } else {
            "held"
        };
        parts.push(format!(
            "{} {:.4} → {:.4} ({delta:+.4}, {verdict}; margin {margin:.4})",
            c.name, bar.score, c.round.score
        ));
    }

    // Every suite is already reported "higher is better" — the assistant suite inverts
    // bits per token for exactly this reason — so one direction word means one thing
    // across the whole sentence. The note used to carry a raw bits/token figure whose
    // unit runs the other way, and a reader took "3.6875 → 3.8796" for progress.
    let detail = parts.join("; ");

    if comparable == 0 {
        return Judgement {
            accept: true,
            note: format!(
                "{detail} — no {bar_label} numbers on record at these scoring versions, \
                 so this round is unmeasured rather than an improvement"
            ),
        };
    }

    let accept = regressed.is_empty() && !improved.is_empty();
    let summary = if !regressed.is_empty() {
        format!(
            "keeping the earlier model: {} regressed beyond its own error bar",
            regressed.join(" and ")
        )
    } else if improved.is_empty() {
        "keeping the earlier model: nothing improved beyond its own error bar".to_string()
    } else {
        format!("accepted on {}", improved.join(" and "))
    };

    Judgement { accept, note: format!("{detail} — {summary}, against {bar_label} numbers") }
}

// The error bar on a difference between two independent runs. NaN if either side
// lacks one, which every caller turns into a documented fallback rather than a zero.
fn stderr_of_difference(a: f64, b: f64) -> f64 {
    (a * a + b * b).sqrt()
}

#[cfg(test)]
mod judge_tests {
    use super::*;
    use db::benchmark::RoundEvidence;

    // The standard errors each suite actually reports at this model size. Language is
    // ~0.95 per run and markup ~1.3 (both measured); the assistant suite's ~2.1 falls
    // out of propagating a ~0.05 bits/token error through the mapping's 41.7 points per
    // bit. A margin is `sqrt(2)` times these: 1.34, 1.84 and 2.97 respectively.
    const LANGUAGE_SE: f64 = 0.95;
    const MARKUP_SE: f64 = 1.3;
    const REPLY_SE: f64 = 2.1;

    // The two scores the assistant suite gives the bits/token pair that started all
    // this. 3.6875 → 42.2505 and 3.8796 → 34.6906: the *same* round, now unmistakably
    // a fall, because the suite inverts the cost rather than leaving the reader to.
    const REPLIES_GOOD: f64 = 42.2505;
    const REPLIES_BAD: f64 = 34.6906;

    fn cmp(name: &'static str, bar: f64, round: f64, stderr: f64) -> SuiteComparison {
        SuiteComparison {
            name,
            round: RoundEvidence { score: round, score_stderr: stderr },
            bar: Some(RoundEvidence { score: bar, score_stderr: stderr }),
        }
    }

    fn language(bar: f64, round: f64) -> SuiteComparison {
        cmp("language", bar, round, LANGUAGE_SE)
    }
    fn markup(bar: f64, round: f64) -> SuiteComparison {
        cmp("markup", bar, round, MARKUP_SE)
    }
    fn replies(bar: f64, round: f64) -> SuiteComparison {
        cmp(ASSISTANT_SUITE, bar, round, REPLY_SE)
    }

    fn judged(suites: Vec<SuiteComparison>) -> Judgement {
        judge(&suites, "the seed's", 0.5)
    }

    // The halt this whole change exists to remove, with the measured numbers: language
    // 58.9434 → 58.3871 is a 0.556 fall against a 1.34 margin, i.e. 0.41σ, while
    // `words` rose and grammar held. If the replies improved, that is a round to keep —
    // the alignment tax is a real property of the trade, not a regression.
    #[test]
    fn the_alignment_tax_alone_does_not_halt_a_cycle() {
        let verdict = judged(vec![
            language(58.9434, 58.3871),
            markup(32.1, 32.1),
            replies(REPLIES_BAD, REPLIES_BAD + 3.8),
        ]);
        assert!(verdict.accept, "{}", verdict.note);
    }

    // The guard has to actually guard, or the cycle trades English for reply fit and
    // nothing ever says so.
    #[test]
    fn losing_english_is_rejected_however_good_the_replies_get() {
        let verdict = judged(vec![
            language(58.9434, 52.0),
            markup(32.1, 32.1),
            replies(REPLIES_BAD, REPLIES_BAD + 10.0),
        ]);
        assert!(!verdict.accept, "{}", verdict.note);
        assert!(verdict.note.contains("language"), "{}", verdict.note);
        assert!(verdict.note.contains("worse"), "{}", verdict.note);
    }

    // The whole reason the gate is a vector test and not a weighted sum. Any blend of
    // these three accepts this round; the constraint refuses it, and names what broke.
    #[test]
    fn a_large_gain_on_one_suite_cannot_pay_for_a_collapse_on_another() {
        let verdict = judged(vec![
            language(58.9434, 58.9434),
            markup(32.1, 25.0),
            replies(REPLIES_BAD, REPLIES_BAD + 20.0),
        ]);
        assert!(!verdict.accept, "{}", verdict.note);
        assert!(verdict.note.contains("markup"), "{}", verdict.note);
    }

    // The round that was read as an improvement and was not. Under the old note it was
    // "3.6875 → 3.8796 ... no gain" and the unit ran backwards; here every suite reads
    // higher-is-better, so the same round says `worse` in the same direction as the
    // other two.
    #[test]
    fn the_round_that_looked_like_progress_reads_as_a_fall() {
        let verdict = judged(vec![
            language(58.9434, 59.0070),
            markup(32.1, 32.1),
            replies(REPLIES_GOOD, REPLIES_BAD),
        ]);
        assert!(!verdict.accept, "{}", verdict.note);
        assert!(verdict.note.contains("replies"), "{}", verdict.note);
        assert!(verdict.note.contains("-7.5599"), "{}", verdict.note);
        assert!(verdict.note.contains("worse"), "{}", verdict.note);
    }

    #[test]
    fn a_round_whose_replies_improved_is_accepted_and_says_so() {
        let verdict = judged(vec![
            language(58.9434, 59.0070),
            markup(32.1, 32.1),
            replies(REPLIES_BAD, REPLIES_GOOD),
        ]);
        assert!(verdict.accept, "{}", verdict.note);
        assert!(verdict.note.contains("+7.5599"), "{}", verdict.note);
        assert!(verdict.note.contains("accepted on replies"), "{}", verdict.note);
    }

    // Movement smaller than the instrument can resolve is not evidence of anything, and
    // a loop that accepts it runs until it drifts. The same margin decides both
    // directions, so this is also what stops the gate ratcheting on noise.
    #[test]
    fn a_gain_inside_the_error_bar_is_not_a_gain() {
        let verdict = judged(vec![
            language(58.9434, 58.9434),
            markup(32.1, 32.1),
            // 1.74 better against a 2.97 margin.
            replies(REPLIES_GOOD, REPLIES_GOOD + 1.74),
        ]);
        assert!(!verdict.accept, "{}", verdict.note);
        assert!(verdict.note.contains("nothing improved"), "{}", verdict.note);
    }

    // Three suites all holding their ground is a round that bought nothing. Accepting
    // it would let the cycle run forever on flat measurements.
    #[test]
    fn holding_every_suite_is_not_progress() {
        let verdict = judged(vec![
            language(58.9434, 58.9434),
            markup(32.1, 32.1),
            replies(REPLIES_GOOD, REPLIES_GOOD),
        ]);
        assert!(!verdict.accept, "{}", verdict.note);
    }

    // A seed benchmarked before the assistant suite existed has no run to compare
    // against. That suite drops out — it must count as neither a pass nor a fail — and
    // the round is decided on what is left.
    #[test]
    fn a_suite_with_no_baseline_drops_out_of_the_test() {
        let dropped = || SuiteComparison {
            name: ASSISTANT_SUITE,
            round: RoundEvidence { score: REPLIES_GOOD, score_stderr: REPLY_SE },
            bar: None,
        };

        let flat = judged(vec![language(58.9434, 58.9434), markup(32.1, 32.1), dropped()]);
        assert!(!flat.accept, "{}", flat.note);
        assert!(flat.note.contains("no comparable"), "{}", flat.note);

        let gained = judged(vec![language(58.9434, 62.0), markup(32.1, 32.1), dropped()]);
        assert!(gained.accept, "{}", gained.note);
    }

    // Every bar at another scoring version reads as absent, which must not read as
    // zero. Accepting is the only option that lets the cycle run, but it is recorded as
    // an absence of evidence rather than as an improvement.
    #[test]
    fn a_bar_that_is_not_on_record_leaves_the_round_unmeasured() {
        let none = |name: &'static str, score: f64, stderr: f64| SuiteComparison {
            name,
            round: RoundEvidence { score, score_stderr: stderr },
            bar: None,
        };
        let verdict = judged(vec![
            none("language", 58.3871, LANGUAGE_SE),
            none("markup", 32.1, MARKUP_SE),
            none(ASSISTANT_SUITE, REPLIES_GOOD, REPLY_SE),
        ]);
        assert!(verdict.accept, "{}", verdict.note);
        assert!(verdict.note.contains("unmeasured"), "{}", verdict.note);
    }

    // `min_gain` may only ever loosen. If it could tighten a margin below the error
    // bar, the 0.41σ halt comes straight back — that is the whole reason it was demoted
    // from a threshold to a floor.
    #[test]
    fn min_gain_can_loosen_a_margin_but_never_tighten_one() {
        let suites = || {
            vec![
                language(58.9434, 58.3871),
                markup(32.1, 32.1),
                replies(REPLIES_BAD, REPLIES_BAD + 3.8),
            ]
        };

        // Far below every suite's error bar: must change nothing.
        let tight = judge(&suites(), "the seed's", 0.005);
        assert!(tight.accept, "{}", tight.note);

        // Far above it: the language fall is still inside the margin, and now so is the
        // reply gain — so the round buys nothing and is refused.
        let loose = judge(&suites(), "the seed's", 10.0);
        assert!(!loose.accept, "{}", loose.note);
        assert!(loose.note.contains("margin 10.0000"), "{}", loose.note);
    }
}

pub fn stage_name(stage: CycleStage) -> &'static str {
    match stage {
        CycleStage::Generating => "catalogue",
        CycleStage::Selftrain => "self-train",
        CycleStage::Preference => "preference",
        CycleStage::Benchmarking => "benchmark",
        CycleStage::Deciding => "decide",
    }
}

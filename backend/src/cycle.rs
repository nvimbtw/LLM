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

use crate::db;
use crate::db::cycle::{CycleStage, CycleStatus};
use crate::db::job::{JobKind, JobStatus};
use crate::gather;
use crate::queue::Queue;
use crate::training::{TrainConfig, TrainMode};

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
            raft_lr: default_raft_lr(),
            kto_lr: default_kto_lr(),
            beta: default_beta(),
            min_gain: default_min_gain(),
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
                let (approved, rejected) = self.catalogue_counts().await?;
                db::cycle::open_round(&self.pool, cycle.id_cycle, cycle.round, approved, rejected)
                    .await
                    .map_err(|e| format!("could not open round: {e}"))?;

                let request = crate::autolabel::AutolabelRequest {
                    id_model: model,
                    candidates: config.candidates,
                    max_prompts: config.max_prompts,
                    temperature: config.temperature,
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
                let (approved, _) = self.catalogue_counts().await?;
                gather::check_raft(approved)?;
                let cfg = self
                    .train_config(
                        TrainMode::Raft,
                        model,
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
                let (approved, rejected) = self.catalogue_counts().await?;
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
                let id_job = self
                    .enqueue(
                        cycle,
                        JobKind::Benchmark,
                        &format!("{} r{}", cycle.label, cycle.round),
                        serde_json::json!({ "id_model": model }),
                        "benchmarking",
                    )
                    .await?;
                self.set_stage(cycle.id_cycle, CycleStage::Benchmarking, Some(id_job))
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
                let (approved, rejected) = self.catalogue_counts().await?;
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

            CycleStage::Benchmarking => {
                let score = match job.ref_id {
                    Some(id_run) => db::benchmark::find_by_id(&self.pool, id_run)
                        .await
                        .map_err(|e| format!("could not read the benchmark run: {e}"))?
                        .and_then(|run| run.score),
                    None => None,
                };
                self.decide(cycle, score).await?;
            }

            CycleStage::Deciding => {}
        }
        Ok(())
    }

    // Accept the round or stop. The only place a cycle decides anything.
    async fn decide(&self, cycle: &db::cycle::Cycle, score: Option<f64>) -> Result<(), String> {
        let config: CycleConfig = serde_json::from_value(cycle.config.clone())
            .map_err(|e| format!("stored cycle config is unreadable: {e}"))?;
        let produced = cycle
            .id_model_current
            .ok_or_else(|| "the round produced no model".to_string())?;

        let Some(score) = score else {
            // Without a score there is no way to tell an improvement from a
            // regression, and continuing would be running blind for hours.
            let note = "the benchmark produced no score, so this round cannot be judged";
            self.close_round(cycle, Some(produced), None, "failed", note)
                .await;
            return Err(note.to_string());
        };

        // What the round has to beat. After round 1 that is the best round so far;
        // for round 1 it is the model the cycle started from, looked up rather than
        // re-benchmarked (the seed is a saved model, so its score is already on
        // record). Treating round 1 as "nothing to beat" is what let all three
        // recorded cycles accept a first round well below their own seed — one of
        // them 6.6 points below, with grammar falling from 0.598 to 0.492, i.e.
        // below chance.
        let (bar, bar_label) = match cycle.best_score {
            Some(best) => (Some(best), "the best round's"),
            None => {
                let seed = match cycle.id_model_start {
                    Some(id) => db::benchmark::latest_score_for_model(&self.pool, id)
                        .await
                        .map_err(|e| format!("could not read the seed model's score: {e}"))?,
                    None => None,
                };
                (seed, "the starting model's")
            }
        };

        // A seed that has never been benchmarked leaves round 1 genuinely
        // unjudgeable. Accepting is still the only option that lets the cycle run,
        // but it is recorded as unmeasured rather than as an improvement.
        let improved = match bar {
            Some(bar) => score > bar + config.min_gain,
            None => true,
        };

        if !improved {
            let note = format!(
                "scored {:.4}, which does not beat {} {:.4} by the {:.4} \
                 required — keeping the earlier model",
                score,
                bar_label,
                bar.unwrap_or(f64::NAN),
                config.min_gain,
            );
            self.close_round(cycle, Some(produced), Some(score), "no_improvement", &note)
                .await;
            self.halt(cycle.id_cycle, &note).await;
            return Ok(());
        }

        let note = format!("scored {score:.4}");
        self.close_round(cycle, Some(produced), Some(score), "accepted", &note)
            .await;
        if let Err(e) = db::cycle::set_best(&self.pool, cycle.id_cycle, produced, score).await {
            eprintln!("[cycle] could not record the best model: {e}");
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
            datasets: Vec::new(),
            curriculum: Vec::new(),
            shuffle: false,
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

    async fn catalogue_counts(&self) -> Result<(i64, i64), String> {
        let counts = db::signal::counts(&self.pool)
            .await
            .map_err(|e| format!("could not count signals: {e}"))?;
        let (auto_up, auto_down) = crate::autolabel::catalogue_counts();
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

pub fn stage_name(stage: CycleStage) -> &'static str {
    match stage {
        CycleStage::Generating => "catalogue",
        CycleStage::Selftrain => "self-train",
        CycleStage::Preference => "preference",
        CycleStage::Benchmarking => "benchmark",
        CycleStage::Deciding => "decide",
    }
}

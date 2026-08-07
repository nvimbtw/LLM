//! The GPU queue and everything that drives it.
//!
//! There is one GPU, so there is one runner and one line: every use of the
//! device — a training run, a benchmark, a chat turn — is a row in `jobs`, and
//! `queue` is the only thing that starts one.
//!
//! **These three are one crate by necessity, not by preference.**
//! `train_worker` reads `queue::{ProgressTick, emit, frame, lock}` and `queue`
//! calls `train_worker::{run, snapshot_frame}` and matches on its `Outcome`.
//! That is a cycle, and Rust has no circular crate dependencies. Splitting them
//! means first extracting the shared progress/telemetry types into a fourth
//! crate — a design change, not a move. `cycle` joins them because it drives the
//! queue across rounds and nothing outside the web layer depends on it.

pub mod chain;
pub mod cycle;
pub mod queue;
pub mod train_worker;

use std::sync::Arc;

use sqlx::PgPool;

use crate::cycle::Cycles;
use crate::queue::Queue;

/// The running drivers, for the HTTP layer to register as shared state.
///
/// Both are `Arc` because a route handler and a socket may reach the same queue
/// from different workers; cloning shares the driver rather than copying it.
pub struct Running {
    pub queue: Arc<Queue>,
    pub cycles: Arc<Cycles>,
}

/// Bring the job system up: reconcile what a dead process left behind, then
/// start the runners and the cycle driver.
///
/// The counterpart of `db::create_pool`, which likewise does more than its name
/// suggests (migrations, admin seeding) because that is the db's own startup
/// ritual. This is the job system's. The entrypoint should not have to know
/// that a crashed run leaves rows to clean up, how many lanes there are, or in
/// what order the two drivers start.
pub async fn start(pool: PgPool) -> Running {
    reconcile_after_restart(&pool).await;

    let queue = Arc::new(Queue::new(pool.clone()));
    // Before the runners start, so a chain is settled before anything can be
    // claimed: a follow-up whose parent finished while this process was dead is
    // promoted here, and one whose parent died mid-run is cancelled here.
    reconcile_chains(&pool, &queue).await;
    queue.spawn();

    // Above the queue: it enqueues the jobs a post-training round is made of and
    // advances when they finish, and it never touches the device itself — a
    // driver that held the GPU would deadlock against its own children. Started
    // after the queue because it subscribes to the queue's event channel.
    let cycles = Arc::new(Cycles::new(pool.clone(), queue.clone()));
    cycles.spawn();

    Running { queue, cycles }
}

/// Close out rows left `running` by a process that died.
///
/// Nothing is in flight at startup by definition: a worker lives on a thread
/// with no durable state, so any row still marked running belongs to a dead
/// process. Left alone they hold the head of the queue forever and show a
/// throbber that never resolves. Queued rows are deliberately untouched — the
/// runners drain them, which is the point of a queue that survives a restart.
///
/// Failure to reconcile is logged, not fatal: a stale row is a bad board, but
/// refusing to boot over it is worse, and the next restart tries again.
async fn reconcile_after_restart(pool: &PgPool) {
    match db::job::fail_orphaned(pool).await {
        Ok(0) => {}
        Ok(n) => {
            println!("closed {} interrupted job(s)", n);
        }
        Err(e) => eprintln!("failed to close interrupted jobs: {}", e),
    }
    // Benchmark history rows are opened by the runner, so the same crash
    // orphans them in the same way.
    match db::benchmark::fail_orphaned(pool).await {
        Ok(0) => {}
        Ok(n) => println!("closed {} interrupted benchmark run(s)", n),
        Err(e) => eprintln!("failed to close interrupted benchmark runs: {}", e),
    }
}

/// Settle follow-ups whose dependency landed while this process was not running.
///
/// The queue resolves a chain the moment a job finishes, which covers every normal
/// completion — but not the window between `db::job::finish` committing and the
/// resolver running, and not a job closed out by `fail_orphaned` above. Both leave
/// a row that waits forever, so the same check runs once at boot, over exactly the
/// rows that need it (`list_orphaned_waiting`).
///
/// Grouped by dependency because `chain::resolve` settles all of one job's children
/// together: five follow-ups behind one dead pretrain is one pass, not five.
async fn reconcile_chains(pool: &PgPool, queue: &Queue) {
    let stranded = match db::job::list_orphaned_waiting(pool).await {
        Ok(stranded) => stranded,
        Err(e) => {
            eprintln!("failed to read stranded follow-up jobs: {}", e);
            return;
        }
    };
    if stranded.is_empty() {
        return;
    }

    let mut settled: Vec<uuid::Uuid> = Vec::new();
    let mut resolution = chain::Resolution::default();

    for job in &stranded {
        // No dependency left to consult: the FK nulled it when the job it followed
        // was deleted. Cancelled with that as the reason.
        let Some(id_parent) = job.depends_on else {
            let one = chain::cancel_orphans(pool, queue, std::slice::from_ref(job)).await;
            resolution.promoted += one.promoted;
            resolution.cancelled += one.cancelled;
            continue;
        };
        if settled.contains(&id_parent) {
            continue;
        }
        settled.push(id_parent);

        match db::job::find_by_id(pool, id_parent).await {
            Ok(Some(parent)) => {
                let one = chain::resolve(pool, queue, &parent).await;
                resolution.promoted += one.promoted;
                resolution.cancelled += one.cancelled;
            }
            // The FK makes this unreachable, so it is a corrupt row rather than a
            // race. Cancel it rather than leave it waiting on nothing.
            Ok(None) => {
                let one = chain::cancel_orphans(pool, queue, std::slice::from_ref(job)).await;
                resolution.cancelled += one.cancelled;
            }
            Err(e) => eprintln!("failed to read job {}: {}", id_parent, e),
        }
    }

    if resolution.touched() {
        println!(
            "settled stranded follow-ups: {} queued, {} cancelled",
            resolution.promoted, resolution.cancelled
        );
    }
}

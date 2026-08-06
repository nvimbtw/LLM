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
        Ok(n) => println!("closed {} interrupted job(s)", n),
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

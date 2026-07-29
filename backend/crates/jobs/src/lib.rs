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

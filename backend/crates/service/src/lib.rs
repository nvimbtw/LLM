//! The boundary to a Python service: the process, and the hop into it.
//!
//! Nothing here names a service. One is a `Config` carrying an audience, an
//! environment prefix and a working directory; `Supervisor` starts it and keeps
//! it alive, and `Proxy` forwards signed requests to it. The crates above —
//! `engine`, `data-engine` — are thin by design: each supplies those constants
//! and a newtype, and adds no behaviour.
//!
//! The newtype is not ceremony. Actix keys `app_data` by type, so two bare
//! `Proxy` values silently overwrite each other and every route reaches whichever
//! service was registered last.
//!
//! `inbound` is the one hop that runs the other way — a service asking the
//! backend for something a *different* service owns, which is how the two Python
//! processes stay unable to reach each other while still sharing data.

pub mod config;
pub mod error;
pub mod inbound;
pub mod proxy;
pub mod supervisor;

pub use config::{Config, Defaults};
pub use error::{Error, Result};
pub use inbound::{InboundSecret, Refused};
pub use proxy::{Frame, Method, Proxy};
pub use supervisor::Supervisor;

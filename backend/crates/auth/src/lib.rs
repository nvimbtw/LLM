//! Who the caller is, and what they may do.
//!
//! Two audiences, two secrets. A token minted for the browser is not accepted
//! by the LLM engine and vice versa, so leaking one secret cannot forge the other.

pub mod access;
pub mod password;
pub mod token;
pub mod token_cipher;

pub use access::AccessLevel;
pub use token::{Caller, Claims, ServiceClaims};

use std::fmt;

#[derive(Debug)]
pub enum EngineError {
    /// The engine could not be reached at all.
    Transport(String),
    /// The engine answered, and refused.
    Status { code: u16, detail: String },
    /// The engine answered with something this build cannot read.
    Decode(String),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EngineError::Transport(e) => write!(f, "engine unreachable: {e}"),
            EngineError::Status { code, detail } => write!(f, "engine returned {code}: {detail}"),
            EngineError::Decode(e) => write!(f, "unreadable engine response: {e}"),
        }
    }
}

impl std::error::Error for EngineError {}

pub type Result<T> = std::result::Result<T, EngineError>;

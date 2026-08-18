use std::fmt;

#[derive(Debug)]
pub enum ToolError {
    /// Nothing is registered under that name. The model asked for a tool this
    /// build does not have, which is a training problem rather than a fault.
    Unknown(String),
    /// Registered, but not usable right now — most often a missing API key.
    Unavailable(&'static str),
    /// The provider could not be reached.
    Transport(String),
    /// The provider answered, and refused.
    Status { code: u16, detail: String },
    /// The provider answered with something this build cannot read.
    Decode(String),
    Timeout,
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::Unknown(name) => write!(f, "no tool named {name}"),
            ToolError::Unavailable(why) => write!(f, "tool unavailable: {why}"),
            ToolError::Transport(e) => write!(f, "tool unreachable: {e}"),
            ToolError::Status { code, detail } => write!(f, "tool returned {code}: {detail}"),
            ToolError::Decode(e) => write!(f, "unreadable tool response: {e}"),
            ToolError::Timeout => write!(f, "tool timed out"),
        }
    }
}

impl std::error::Error for ToolError {}

pub type Result<T> = std::result::Result<T, ToolError>;

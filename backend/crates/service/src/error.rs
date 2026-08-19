use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// The service could not be reached at all.
    Transport(String),
    /// The service answered, and refused.
    Status { code: u16, detail: String },
    /// The service answered with something this build cannot read.
    Decode(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Transport(e) => write!(f, "service unreachable: {e}"),
            Error::Status { code, detail } => write!(f, "service returned {code}: {detail}"),
            Error::Decode(e) => write!(f, "unreadable service response: {e}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

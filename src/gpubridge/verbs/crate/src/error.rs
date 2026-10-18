use std::fmt;

use crate::types::QpState;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A verbs call failed with this errno.
    Verbs { call: &'static str, errno: i32 },
    /// Bad arguments, refused before any verbs call.
    InvalidArgument(String),
    /// A queue pair state change verbs doesn't allow, or a call the state doesn't allow.
    InvalidState {
        state: QpState,
        wanted: &'static str,
    },
    /// More outstanding work than the queue pair was created for.
    QueueFull { queue: &'static str, limit: u32 },
    /// A limit reached: the cache's bounds, a completion queue's capacity, a device's.
    Exhausted(String),
    /// The resource has work outstanding and can't do this now.
    Busy(String),
    /// A queue pair's work didn't finish draining in time; the memory it references was
    /// leaked rather than released.
    DrainTimeout { qp_num: u32, outstanding: usize },
    /// The provider can't do this (the mock provider and dma-buf, for example).
    Unsupported(String),
}

impl Error {
    pub(crate) fn invalid(text: impl Into<String>) -> Error {
        Error::InvalidArgument(text.into())
    }

    /// The errno, for a failed verbs call.
    pub fn errno(&self) -> Option<i32> {
        match self {
            Error::Verbs { errno, .. } => Some(*errno),
            _ => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Verbs { call, errno } => write!(
                f,
                "{call} failed: {} (errno {errno})",
                std::io::Error::from_raw_os_error(*errno)
            ),
            Error::InvalidArgument(text) => write!(f, "invalid argument: {text}"),
            Error::InvalidState { state, wanted } => {
                write!(f, "queue pair in state {state:?} can't {wanted}")
            }
            Error::QueueFull { queue, limit } => {
                write!(f, "{queue} queue full ({limit} outstanding)")
            }
            Error::Exhausted(text) => write!(f, "exhausted: {text}"),
            Error::Busy(text) => write!(f, "busy: {text}"),
            Error::DrainTimeout {
                qp_num,
                outstanding,
            } => write!(
                f,
                "queue pair {qp_num:#x}: {outstanding} work requests still outstanding at the \
                 drain deadline; their memory regions are leaked, not released"
            ),
            Error::Unsupported(text) => write!(f, "unsupported: {text}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

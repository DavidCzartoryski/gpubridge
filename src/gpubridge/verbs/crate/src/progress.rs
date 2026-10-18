//! Native progress policy and optional Linux affinity. Policy is local, not wire state.

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Progress {
    Latency,
    Balanced,
    /// The original drain/arm/re-poll/event-wait behavior.
    #[default]
    Economy,
}

impl Progress {
    pub fn parse(text: &str) -> Result<Self> {
        match text {
            "latency" => Ok(Self::Latency),
            "balanced" => Ok(Self::Balanced),
            "economy" => Ok(Self::Economy),
            _ => Err(Error::invalid(
                "progress must be latency, balanced or economy",
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Latency => "latency",
            Self::Balanced => "balanced",
            Self::Economy => "economy",
        }
    }
}

/// Validates against the calling thread's inherited allowed CPU set before applying
/// affinity to that thread only. Never expands the process/job CPU allocation.
pub(crate) fn cpu(cpu: u32, apply: bool) -> Result<()> {
    #[cfg(feature = "ibverbs")]
    {
        extern "C" {
            fn gbv_progress_cpu(cpu: u32, apply: i32) -> i32;
        }
        // SAFETY: only scalar arguments; the header-compiled shim owns cpu_set_t's
        // platform layout and checks CPU membership before sched_setaffinity(0, ...).
        let rc = unsafe { gbv_progress_cpu(cpu, i32::from(apply)) };
        if rc == 0 {
            Ok(())
        } else {
            Err(Error::invalid(format!(
                "progress_cpu {cpu} is not usable in the thread's allowed CPU set: {}",
                std::io::Error::from_raw_os_error(rc)
            )))
        }
    }
    #[cfg(not(feature = "ibverbs"))]
    {
        let _ = (cpu, apply);
        Err(Error::Unsupported(
            "progress_cpu requires the Linux ibverbs build".into(),
        ))
    }
}

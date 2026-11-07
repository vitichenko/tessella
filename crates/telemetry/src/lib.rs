#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::significant_drop_tightening)]
#![allow(clippy::format_push_string)]
#![allow(clippy::uninlined_format_args)]

pub mod logging;

pub use logging::init_logging;

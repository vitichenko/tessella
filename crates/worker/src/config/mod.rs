//! Worker configuration module (SOLID-91).
//!
//! Submodules:
//! - [`schema`]: Core configuration structures, serialization, and deserialization.
//! - [`env`]: Environment variable overrides, field precedence rules, and parser helpers.
//! - [`validate`]: Semantic invariant validation and deprecation warnings.
//! - [`derived`]: Computed defaults, device resolution, and helper methods.

pub mod derived;
pub mod env;
pub mod schema;
pub mod validate;

pub use derived::*;
pub use env::*;
pub use schema::*;
pub use validate::*;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_compat;

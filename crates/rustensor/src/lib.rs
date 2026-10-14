//! A small, dependency-free CPU tensor foundation.
mod error;
mod tensor;
pub use error::{Error, Result};
pub use tensor::{broadcast_shape, Tensor};

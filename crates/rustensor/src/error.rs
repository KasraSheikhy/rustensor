use std::fmt;
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    ShapeOverflow,
    ShapeMismatch,
    InvalidIndex,
    InvalidMatmul,
}
pub type Result<T> = std::result::Result<T, Error>;
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ShapeOverflow => "shape exceeds addressable size",
            Self::ShapeMismatch => "shape and data or operand shapes do not match",
            Self::InvalidIndex => "index rank or coordinate is invalid",
            Self::InvalidMatmul => "matmul requires compatible rank-two tensors",
        })
    }
}
impl std::error::Error for Error {}

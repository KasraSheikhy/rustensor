use crate::{Error, Result};

/// Owned contiguous row-major f64 tensor. Empty shape denotes a scalar.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f64>,
    shape: Vec<usize>,
}
fn numel(shape: &[usize]) -> Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d).ok_or(Error::ShapeOverflow))
}
impl Tensor {
    pub fn from_vec(data: Vec<f64>, shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = shape.into();
        if numel(&shape)? != data.len() {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self { data, shape })
    }
    pub fn zeros(shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = shape.into();
        let n = numel(&shape)?;
        Self::from_vec(vec![0.0; n], shape)
    }
    pub fn scalar(value: f64) -> Self {
        Self {
            data: vec![value],
            shape: vec![],
        }
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn data(&self) -> &[f64] {
        &self.data
    }
    pub fn get(&self, index: &[usize]) -> Result<f64> {
        if index.len() != self.shape.len() {
            return Err(Error::InvalidIndex);
        }
        let mut offset = 0;
        for (&i, &d) in index.iter().zip(&self.shape) {
            if i >= d {
                return Err(Error::InvalidIndex);
            }
            offset = offset * d + i;
        }
        Ok(self.data[offset])
    }
    /// Reshape consumes the tensor and reuses its allocation.
    pub fn reshape(mut self, shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape = shape.into();
        if numel(&shape)? != self.data.len() {
            return Err(Error::ShapeMismatch);
        }
        self.shape = shape;
        Ok(self)
    }
    pub fn map(&self, f: impl Fn(f64) -> f64) -> Self {
        Self {
            data: self.data.iter().copied().map(f).collect(),
            shape: self.shape.clone(),
        }
    }
    /// Elementwise addition with right-aligned broadcasting.
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, |a, b| a + b)
    }
    /// Elementwise multiplication with right-aligned broadcasting.
    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        self.binary(rhs, |a, b| a * b)
    }
    fn binary(&self, rhs: &Self, op: impl Fn(f64, f64) -> f64) -> Result<Self> {
        let shape = broadcast_shape(&self.shape, &rhs.shape)?;
        let count = numel(&shape)?;
        let mut data = Vec::with_capacity(count);
        // Empty outputs never compute strides or read input storage.
        for flat in 0..count {
            let left = broadcast_offset(flat, &shape, &self.shape);
            let right = broadcast_offset(flat, &shape, &rhs.shape);
            data.push(op(self.data[left], rhs.data[right]));
        }
        Self::from_vec(data, shape)
    }
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        if self.shape.len() != 2 || rhs.shape.len() != 2 || self.shape[1] != rhs.shape[0] {
            return Err(Error::InvalidMatmul);
        }
        let (m, k, n) = (self.shape[0], self.shape[1], rhs.shape[1]);
        let mut out = Self::zeros(vec![m, n])?;
        if m == 0 || n == 0 || k == 0 {
            return Ok(out);
        }
        for i in 0..m {
            for t in 0..k {
                for j in 0..n {
                    out.data[i * n + j] += self.data[i * k + t] * rhs.data[t * n + j];
                }
            }
        }
        Ok(out)
    }
}

/// Resolve compatible shapes without allocating expanded operand storage.
/// Missing leading dimensions and dimensions of size one broadcast.
pub fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>> {
    let rank = left.len().max(right.len());
    let mut shape = vec![1; rank];
    for (axis, out) in shape.iter_mut().enumerate() {
        let l = axis.checked_sub(rank - left.len()).map_or(1, |i| left[i]);
        let r = axis.checked_sub(rank - right.len()).map_or(1, |i| right[i]);
        *out = if l == r || r == 1 {
            l
        } else if l == 1 {
            r
        } else {
            return Err(Error::ShapeMismatch);
        };
    }
    numel(&shape)?;
    Ok(shape)
}

// Decode an output flat index and map broadcast dimensions to coordinate zero.
fn broadcast_offset(mut flat: usize, output: &[usize], input: &[usize]) -> usize {
    let leading = output.len() - input.len();
    let mut offset = 0;
    let mut stride = 1;
    for axis in (0..output.len()).rev() {
        let coordinate = flat % output[axis];
        flat /= output[axis];
        if axis >= leading {
            let dim = input[axis - leading];
            if dim != 1 {
                offset += coordinate * stride;
            }
            stride *= dim;
        }
    }
    offset
}

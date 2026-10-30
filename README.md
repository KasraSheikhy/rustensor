# Rustensor

Numerical computing from scratch in Rust.

The project is still small. Right now it has a CPU tensor core; the longer-term goal is to build out array operations and then automatic differentiation.

## What works so far

- Tensors backed by a `Vec<f64>`, stored in row-major order.
- Construction from a vector, zeros, and scalars.
- Indexing, reshape, and elementwise mapping.
- Addition and multiplication with broadcasting.
- Sum over all elements.
- Matrix multiplication for two-dimensional tensors.
- Validation for shapes, indices, and shape overflow.

An empty shape (`[]`) represents a scalar. Zero-sized dimensions are supported. `reshape` consumes the tensor and keeps its existing allocation.

There is no autograd or GPU support yet, and the API will change as the project grows.

## Build and test

With Rust and Cargo installed:

```sh
git clone https://github.com/KasraSheikhy/rustensor.git
cd rustensor
cargo build --workspace
cargo test --workspace
```

The tensor core currently has no external dependencies. It is not published to crates.io yet.

## Example

Save this as `crates/rustensor/examples/basic.rs` (create the `examples` directory first):

```rust
use rustensor::{Result, Tensor};

fn main() -> Result<()> {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2])?;
    let row = Tensor::from_vec(vec![10.0, 20.0], vec![2])?;

    // The row is added to each row of a.
    let added = a.add(&row)?;
    assert_eq!(added.data(), &[11.0, 22.0, 13.0, 24.0]);

    let weights = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1])?;
    let product = a.matmul(&weights)?;
    assert_eq!(product.shape(), &[2, 1]);
    assert_eq!(product.data(), &[5.0, 11.0]);

    assert_eq!(a.get(&[1, 0])?, 3.0);
    assert_eq!(a.sum(), 10.0);

    let flat = a.reshape(vec![4])?;
    assert_eq!(flat.shape(), &[4]);

    Ok(())
}
```

Run it from the repository root:

```sh
cargo run -p rustensor --example basic
```

## Next steps

- Subtraction, division, and more elementwise operations.
- Reductions along an axis, starting with sum and mean.
- Transpose and basic slicing.
- A linear-regression example using manual gradients.
- Automatic differentiation once the tensor operations are in place.

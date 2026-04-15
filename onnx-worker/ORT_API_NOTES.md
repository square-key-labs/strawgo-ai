# ORT v2 API Notes

## Working Cargo.toml

```toml
[package]
name = "onnx-worker"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "onnx-worker"
path = "src/main.rs"

[dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros", "net", "io-util"] }
anyhow = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
ort = { version = "2.0.0-rc.12", features = ["download-binaries"] }
rustfft = "6"
rubato = "0.15"
ndarray = "0.17"
```

**Key notes:**
- Use the explicit RC version `"2.0.0-rc.12"` — `"2"` will not resolve pre-releases.
- `download-binaries` is the correct feature name (it is also in ort's own default features).
- Use `ndarray = "0.17"` to match ort's transitive dependency; using `0.16` would produce two incompatible ndarray versions.
- Build downloads the ONNX Runtime native binary at compile time — first build is slow (expected).

---

## Session Builder API (v2)

Load a model from a file path:

```rust
use ort::session::{Session, builder::GraphOptimizationLevel};

let session = Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .with_intra_threads(1)?
    .commit_from_file("path/to/model.onnx")?;
```

Load from a URL (useful for examples/tests):

```rust
let session = Session::builder()?
    .commit_from_url("https://example.com/model.onnx")?;
```

`Session::builder()` returns `ort::Result<SessionBuilder>`. Each builder method also returns `ort::Result<SessionBuilder>`. `commit_from_file` / `commit_from_url` return `ort::Result<Session>`.

---

## Tensor Creation API (v2)

ort v2 uses `TensorRef` for zero-copy views or owned `Tensor` values.

### f32 tensor from a Vec<f32> with given shape

```rust
use ort::value::TensorRef;

let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
let shape = [1_usize, 4_usize]; // or vec![1i64, 4i64] — usize or i64 slices both work

let tensor = TensorRef::from_array_view((shape, data.as_slice()))?;
```

For a 2-D shape (batch × length):

```rust
let tensor = TensorRef::from_array_view(([batch_size, seq_len], data.as_slice()))?;
```

### i64 tensor from a Vec<i64>

```rust
let ids: Vec<i64> = vec![101, 7592, 102];
let shape = [1_usize, ids.len()];

let tensor = TensorRef::from_array_view((shape, ids.as_slice()))?;
```

`TensorRef::from_array_view` takes a tuple `(shape, &[T])` where `shape` can be any type that implements `ort`'s `Shape` trait — `[usize; N]`, `Vec<usize>`, `Vec<i64>`, etc.

---

## Running Inference

```rust
use ort::inputs;

// Named inputs (by position — order matters)
let outputs = session.run(inputs![tensor_a, tensor_b])?;

// Named inputs (by name)
let outputs = session.run(inputs!["input_ids" => tensor_ids, "attention_mask" => tensor_mask])?;
```

`session.run(...)` returns `ort::Result<SessionOutputs>`. `SessionOutputs` implements `Index<&str>` and `Index<usize>`.

---

## Extracting Output Tensor Data

### As a flat slice (lowest-overhead)

```rust
// Returns (shape_dims, &[f32])
let (dim, data) = outputs["output_name"].try_extract_tensor::<f32>()?;
// dim: Vec<i64>  — e.g. [1, 128, 768]
// data: &[f32]   — row-major flat slice
```

### As an ndarray::ArrayView (requires ndarray feature, enabled by default)

```rust
use ndarray::{Ix2, Ix3};

// Dynamically dimensioned
let arr = outputs[0].try_extract_array::<f32>()?;

// Cast to a known rank (e.g. 2-D)
let arr_2d = arr.into_dimensionality::<Ix2>()?;
```

---

## Complete Minimal Example

```rust
use ort::{
    inputs,
    session::Session,
    value::TensorRef,
};

fn run_model(model_path: &str, input_data: &[f32], shape: &[usize]) -> ort::Result<Vec<f32>> {
    let session = Session::builder()?.commit_from_file(model_path)?;

    let tensor = TensorRef::from_array_view((shape, input_data))?;
    let outputs = session.run(inputs![tensor])?;

    let (_dims, values) = outputs[0].try_extract_tensor::<f32>()?;
    Ok(values.to_vec())
}
```

---

## Version Reference

| Crate | Version used | Notes |
|-------|-------------|-------|
| ort | 2.0.0-rc.12 | Latest RC as of 2026-04 |
| ort-sys (transitive) | 2.0.0-rc.12 | Bundles/downloads ORT 1.24.x |
| ndarray | 0.17.2 | Must match ort's requirement |
| rustfft | 6.4.1 | — |
| rubato | 0.15.0 | — |

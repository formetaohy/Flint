# Thuban

GPU-accelerated LLM inference engine built on portable compute shaders.

## Usage

```toml
[dependencies]
thuban = "0.1.0"
```

Every sub-crate is re-exported as a module and gated behind a feature of the
same name, all enabled by default except `thuban_server`:

```rust
use thuban::gpu;
use thuban::architectures;
```

For a lean build, disable the default features and pick only what you need:

```toml
[dependencies]
thuban = { version = "0.1.0", default-features = false, features = ["thuban_architectures"] }
```

| Feature | Module |
| --- | --- |
| `thuban_architectures` | `architectures` |
| `thuban_backend` | `backend` |
| `thuban_checkpoint` | `checkpoint` |
| `thuban_error` | `error` |
| `thuban_fetch` | `fetch` |
| `thuban_generate` | `generate` |
| `thuban_gpu` | `gpu` |
| `thuban_kernel` | `kernel` |
| `thuban_model` | `model` |
| `thuban_num` | `num` |
| `thuban_profiler` | `profiler` |
| `thuban_server` | `server` | opt-in: `features = ["thuban_server"]` |
| `thuban_tensor` | `tensor` |
| `thuban_tokenizer` | `tokenizer` |

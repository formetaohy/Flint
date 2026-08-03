<!-- <p align="center">
  <img src="assets/logo.png" alt="Flint" width="96" />
</p> -->

<h3 align="center"><strong>Flint</strong></h3>

<p align="center">
  <strong>Fast, cross-platform LLM inference engine.</strong>
</p>

<p align="center">
  <a href="https://github.com/formetaohy/Flint/actions/workflows/ci.yml"><img src="https://github.com/formetaohy/Flint/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/formetaohy/Flint/stargazers"><img src="https://img.shields.io/github/stars/formetaohy/Flint?style=flat-square&color=yellow" alt="Stars" /></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/language-Rust-dea584?style=flat-square&logo=rust" alt="Rust" /></a>
  <a href="https://github.com/formetaohy/Flint/blob/main/LICENSE"><img src="https://img.shields.io/github/license/formetaohy/Flint?style=flat-square&color=blue" alt="License" /></a>
</p>

<p align="center">
  <a href="README_CN.md">中文</a>
</p>

## What is Flint?

Flint is a local LLM inference engine written in pure Rust on [WGPU](https://github.com/gfx-rs/wgpu). It runs on the GPU via portable compute shaders, supporting Windows, Linux/Android, macOS/iOS, and the Web (WASM).

## Features

- **Cross-platform** — Built on WGPU; runs on every platform. 
- **Fast** — Highly optimized WGSL-based inference core.
- **Multi-format** — Loads `safetensors` and quantized `GGUF` models.

## Supported Architectures

- Qwen3.5, Qwen3, Qwen2
- Gemma 3

We're actively adding support for more models. If there's a model you'd like supported, feel free to open an issue or PR.

## Performance

`flint-bench` measures throughput on synthetic weights at real model
dimensions (no downloads):

```sh
cargo run --release -p flint-bench
```

Reference numbers on an RTX 5070 (Llama-8B class: hidden 4096, intermediate
14336, 8 layers):

| Phase | Before | After | Speedup |
| --- | --- | --- | --- |
| prefill | 200 tok/s | 510 tok/s | 2.5x |
| decode (2K context) | 67 tok/s | 149 tok/s | 2.2x |

Key optimizations, mapped to published methods:

- **Streaming gemm/gemv** (SplitK work decomposition, arXiv:2402.00025):
  decode splits K across workgroups; narrow projections (q/k/v) go from
  45 GB/s to 216+ GB/s. Scalar register accumulators avoid local-memory
  spills.
- **Split-K grouped-query attention** (FlashAttention, FlashInfer,
  arXiv:2501.01005; PAT, arXiv:2511.22333): the KV range is split into
  parallel segments (8 -> 256 workgroups during decode) and query heads of
  the same kv head share staged K/V tiles; long-context attention is ~8x
  faster.
- **bf16 KV cache**: halves runtime memory, doubles attention bandwidth.

## Quick Start

Requires [Rust](https://rustup.rs).

```sh
git clone https://github.com/formetaohy/Flint.git

cd Flint

# Point --model at any supported checkpoint directory (safetensors or GGUF).
cargo run --release -- --model /path/to/Qwen3.5-0.8B --prompt "What is a tensor?"
```

## Contact

Email: **formetaohy@gmail.com**

## License

[GPL-3.0](LICENSE)

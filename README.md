<p align="center">
  <!-- TODO: replace with the real logo -->
  <img src="assets/logo.png" alt="Flint" width="96" />
</p>

<h3 align="center"><strong>Flint</strong></h3>

<p align="center">
  <strong>A general, fast LLM inference engine.<br>Built in Rust. Runs everywhere, no CUDA required.</strong>
</p>

<p align="center">
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
- **Broadly compatible** — Supports `safetensors` and quantized `GGUF` formats.

## Supported Architectures

- Qwen3.5, Qwen3, Qwen2
- Gemma 3

We're actively adding support for more models. If there's a model you'd like supported, feel free to open an issue or PR.

## Getting Started

Requires [Rust](https://rustup.rs) and [huggingface cli](https://huggingface.co/docs/huggingface_hub/guides/cli).

```sh
git clone https://github.com/formetaohy/Flint.git

cd Flint

hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B

cargo run --release -- --model models/Qwen3.5-0.8B --prompt "What is a tensor?"
```

## Contact

Email: **formetaohy@gmail.com**

## License

[GPL-3.0](LICENSE)

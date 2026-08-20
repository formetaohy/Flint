![LOGO](assets/logo.svg)

<p align="center">
  <strong>Fast, cross-platform LLM inference engine.</strong>
</p>

<p align="center">
  <a href="https://github.com/formetaohy/Flint/actions/workflows/ci.yml"><img src="https://github.com/formetaohy/Flint/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/formetaohy/Flint/stargazers"><img src="https://img.shields.io/github/stars/formetaohy/Flint?style=flat-square&color=yellow" alt="Stars" /></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/language-Rust-dea584?style=flat-square&logo=rust" alt="Rust" /></a>
  <a href="https://github.com/formetaohy/Flint/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License" /></a>
</p>

<p align="center">
  <a href="README_CN.md">中文</a>
</p>

## What is Flint?

Flint is a LLM inference engine written in pure Rust on [WGPU](https://github.com/gfx-rs/wgpu). It runs on the GPU via portable compute shaders, supporting **Windows**, **Linux/Android**, **macOS/iOS**, and the **Web (WASM)**.

## Features

- **Cross-platform** — Write once, run everywhere.
- **Fast** — Highly optimized inference core.
- **Multi-format** — Supports `safetensors` and `GGUF` formats

## Supported Architectures

| Family | Models
| --- | ---
| Qwen | Qwen3.5 / Qwen3 / Qwen2 / Qwen1.5
| LLaMA | LLaMA 2/3.x, Mistral, Phi-3/3.5
| Gemma | Gemma 2 / 3
| Phi | Phi-4-mini / Phi-3.x
| Phi-MoE | Phi-tiny/mini-MoE
| Gemma 4 | Gemma 4 E2B/E4B

We're actively adding support for more models. If there's a model you'd like supported, feel free to open an issue or PR.


## Quick Start

[Examples](examples/README.md): Flint's dedicated, runnable examples — a great way to get hands-on with the engine.

## Contact

Email: **formetaohy@gmail.com**

## License

[MIT](LICENSE)

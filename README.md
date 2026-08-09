<!-- <p align="center">
  <img src="assets/logo.png" alt="Flint" width="96" />
</p> -->

<h1 align="center"><strong>Flint</strong></h1>

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

**Flint** is a high-performance LLM inference engine written in Rust. At its core lies **Saturn**, a GPU compute abstraction layer purpose-built for cross-platform compute shaders, enabling Flint to run seamlessly on Windows, Linux, Android, macOS, and iOS.

### What is Saturn?

**Saturn** is an abstraction layer designed for cross-platform GPU computing. It adopts compute shaders as a unified computing model and supports both Vulkan and Metal backends. To abstract away platform-specific shader syntax differences, **Saturn** introduces **SCL** (Saturn Compute Language), a cross-platform compute language. 

```scl
kernel add [workgroup(256, 1, 1)] (a: buf<f32>, b: buf<f32>, y: buf<f32>, N_ELEM: u32) {
    if gid.x < N_ELEM {
        y[gid.x] = a[gid.x] + b[gid.x];
    }
}
```

## Features

- **Cross-platform** — Write once, run everywhere.
- **Fast** — Highly optimized inference core.
- **Multi-format** — Supports `safetensors`, `ONNX` and quantized `GGUF` formats

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

[MIT](LICENSE)

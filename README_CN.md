<!-- <p align="center">
  <img src="assets/logo.png" alt="Flint" width="96" />
</p> -->

<h3 align="center">Flint</h3>

<p align="center">
  <strong>高性能、跨平台的 LLM 推理引擎</strong>
</p>

<p align="center">
  <a href="https://github.com/formetaohy/Flint/actions/workflows/ci.yml"><img src="https://github.com/formetaohy/Flint/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/formetaohy/Flint/stargazers"><img src="https://img.shields.io/github/stars/formetaohy/Flint?style=flat-square&color=yellow" alt="Stars" /></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/language-Rust-dea584?style=flat-square&logo=rust" alt="Rust" /></a>
  <a href="https://github.com/formetaohy/Flint/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License" /></a>
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

## Flint 是什么？

Flint 是用纯 Rust 编写、基于 [WGPU](https://github.com/gfx-rs/wgpu) 的本地大模型推理引擎。通过可移植的计算着色器在 GPU 上运行，支持 Windows、Linux/Android、macOS/iOS 与 Web (WASM)。

## 特性

- **跨平台** —— 基于 WGPU 开发，可运行于所有支持的平台。
- **高性能** —— 高度优化的、基于 WGSL 的推理核心。
- **多格式** —— 支持 `safetensors`、`ONNX` 和量化 `GGUF` 格式。

## 支持的架构

| 家族 | 模型
| --- | ---
| Qwen | Qwen3.5 / Qwen3 / Qwen2 / Qwen1.5
| LLaMA | LLaMA 2/3.x、Mistral、Phi-3/3.5
| Gemma | Gemma 2 / 3
| Phi | Phi-4-mini / Phi-3.x
| Phi-MoE | Phi-tiny/mini-MoE
| Gemma 4 | Gemma 4 E2B/E4B

我们正在努力添加对更多模型的支持。如果你有想要支持的模型，欢迎提 issue 或 PR。


## 快速开始

需要 [Rust](https://rustup.rs) 和 [huggingface cli](https://huggingface.co/docs/huggingface_hub/guides/cli).

```sh
git clone https://github.com/formetaohy/Flint.git

cd Flint

hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B

cargo run --release -- --model models/Qwen3.5-0.8B --prompt "What is a tensor?"
```

## 联系

邮箱：**formetaohy@gmail.com**

## 许可证

[MIT](LICENSE)

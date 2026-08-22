<!-- ![LOGO](assets/logo.svg) -->

<p align="center">
  <strong>高性能、跨平台的 LLM 推理引擎</strong>
</p>

<p align="center">
  <a href="https://github.com/formetaohy/Thuban/actions/workflows/ci.yml"><img src="https://github.com/formetaohy/Thuban/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://crates.io/crates/thuban"><img src="https://img.shields.io/crates/v/thuban?style=flat-square" alt="crates.io" /></a>
  <a href="https://github.com/formetaohy/Thuban/stargazers"><img src="https://img.shields.io/github/stars/formetaohy/Thuban?style=flat-square&color=yellow" alt="Stars" /></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/language-Rust-dea584?style=flat-square&logo=rust" alt="Rust" /></a>
  <a href="https://github.com/formetaohy/Thuban/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License" /></a>
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

## Thuban 是什么？

Thuban 是用纯 Rust 编写、基于 [WGPU](https://github.com/gfx-rs/wgpu) 的大模型推理引擎。通过可移植的 Compute Shader 在 GPU 上运行，支持 **Windows**、**Linux/Android**、**macOS/iOS** 和 **Web (WASM)**。

## 特性

- **跨平台** —— 一次编写，随处运行。
- **高性能** —— 高度优化的的推理核心。
- **原生 GGUF 支持** —— 原生支持 GGUF 的所有量化格式。
- **API 服务器** —— 以 OpenAI（chat completions / responses）、Anthropic messages、Gemini API 方式对外提供模型服务

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

[示例](examples/README.md)：Thuban 专门设计的可运行示例，可以帮助你快速上手引擎。

## 联系

邮箱：**formetaohy@gmail.com**

## 许可证

[MIT](LICENSE)

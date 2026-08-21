# chat

A CLI example that downloads a Hugging Face model on demand and runs streaming chat inference with the Thuban compute-shader stack. For thinking models (Qwen3, Qwen3.5, DeepSeek-R1, ...) the stream includes the raw `<think>...</think>` reasoning before the final answer; use the [server](../server/README.md) example if you need reasoning split out per protocol.

## Usage

```sh
cargo run -p thuban_examples --example chat -- --model Qwen/Qwen2.5-0.5B-Instruct --prompt "What is a tensor?" --format safetensors
```

The model must ship GGUF quantizations; if the repo contains several, the largest one (highest quality) is picked automatically.

## Caching

Assets are stored under `temp/<org>--<repo>/` and reused on later runs; a file is re-downloaded only when its size differs from the repo listing.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--max-tokens` | 8192 | Generation limit |
| `--ctx-size` | 32768 | Context length (chat formats) |

## Environment

- `HF_ENDPOINT` — override the Hub base URL (e.g. a mirror)

# chat

A CLI example that downloads a Hugging Face model on demand and runs inference with the Flint compute-shader stack.

## Usage

```sh
cargo run -p chat -- --model Qwen/Qwen2.5-0.5B-Instruct --prompt "What is tensor?" --format safetensors
```

`--format` accepts:

- `safetensors` — decodes via `config.json` + `*.safetensors` shards
- `gguf` — single-file checkpoint; if the repo ships several quantizations, the largest one (highest quality) is picked
- `onnx` — runs through `flint-onnx`; the repo must ship a `tokenizer.json`

## Caching

Assets are stored under `temp/<org>--<repo>/` and reused on later runs; a file is re-downloaded only when its size differs from the repo listing.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--max-tokens` | 8192 | Generation limit |
| `--ctx-size` | 32768 | Context length (chat formats) |

## Environment

- `HF_ENDPOINT` — override the Hub base URL (e.g. a mirror)
- `HTTP_PROXY` / `HTTPS_PROXY` — honored by the HTTP client

# embed

A CLI example that downloads a safetensors embedding model on demand and prints the embedding of a prompt.

## Usage

```sh
cargo run -p flint-examples --example embed -- --model BAAI/bge-base-en-v1.5 --prompt "Hello world"
```

The repo must ship a `tokenizer.json`. Prints the first few values and the total size of the embedding.

## Caching

Assets are stored under `temp/<org>--<repo>/` and reused on later runs; a file is re-downloaded only when its size differs from the repo listing.

## Environment

- `HF_ENDPOINT` — override the Hub base URL (e.g. a mirror)

# schema

A CLI example that downloads a Hugging Face model on demand and generates JSON constrained by a JSON Schema.

## Usage

```sh
cargo run -p flint-examples --example schema -- --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Describe the movie Inception" --schema schema.json
```

`--format` accepts:

- `safetensors` — decodes via `config.json` + `*.safetensors` shards
- `gguf` — single-file checkpoint; if the repo ships several quantizations, the largest one (highest quality) is picked

`--schema` accepts the JSON Schema subset supported by `flint-generate::Grammar`: objects with required and optional properties, arrays, strings, integers, numbers, booleans, null, `enum`, `const`, `anyOf` / `oneOf`, and `$defs` references. Output is guaranteed to be a valid prefix of the schema from the first token on.

## Caching

Assets are stored under `temp/<org>--<repo>/` and reused on later runs; a file is re-downloaded only when its size differs from the repo listing.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--max-tokens` | 8192 | Generation limit |
| `--ctx-size` | 32768 | Context length (chat formats) |

## Environment

- `HF_ENDPOINT` — override the Hub base URL (e.g. a mirror)

# schema

A CLI example that downloads a Hugging Face model on demand and generates JSON constrained by a JSON Schema.

## Usage

```sh
cargo run -p thuban_examples --example schema -- --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Describe the movie Inception" --schema schema.json
```

The model must ship GGUF quantizations; if the repo contains several, the largest one (highest quality) is picked automatically.

`--schema` accepts the JSON Schema subset supported by `thuban_generate::Grammar`: objects with required and optional properties, arrays, strings, integers, numbers, booleans, null, `enum`, `const`, `anyOf` / `oneOf`, and `$defs` references. Output is guaranteed to be a valid prefix of the schema from the first token on. Because the grammar constrains every token from the start, thinking models do not reason in this mode — the model answers directly in JSON.

## Caching

Assets are stored under `temp/<org>--<repo>/` and reused on later runs; a file is re-downloaded only when its size differs from the repo listing.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--max-tokens` | 8192 | Generation limit |
| `--ctx-size` | 32768 | Context length (chat formats) |

## Environment

- `HF_ENDPOINT` — override the Hub base URL (e.g. a mirror)

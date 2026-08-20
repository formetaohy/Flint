# Examples

| Example | Description |
| --- | --- |
| [chat](chat/main.rs) | Chat inference with a chat-tuned model (safetensors or GGUF) |
| [embed](embed/main.rs) | Text embeddings with a BERT-style model |
| [schema](schema/main.rs) | Structured generation constrained by a JSON Schema |

## Usage

```sh
cargo run -p flint-examples --example chat -- --model Qwen/Qwen2.5-0.5B-Instruct --prompt "What is a tensor?"
cargo run -p flint-examples --example embed -- --model BAAI/bge-base-en-v1.5 --prompt "Hello world"
cargo run -p flint-examples --example schema -- --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Describe the movie Inception" --schema schema.json
```

### chat

```
--model       Hugging Face repo id
--prompt      user message
--format      checkpoint format: gguf (default) or safetensors
--max-tokens  output token budget (default 8192)
--ctx-size    KV cache slot length (default 32768)
```

### embed

```
--model       Hugging Face repo id of a safetensors BERT model
--prompt      text to embed
```

Prints the first few values and the total size of the L2-normalized embedding.

### schema

```
--model       Hugging Face repo id
--prompt      instruction describing the desired JSON
--schema      path to a JSON Schema file
--format      checkpoint format: gguf (default) or safetensors
--max-tokens  output token budget (default 8192)
--ctx-size    KV cache slot length (default 32768)
```

`--schema` accepts the JSON Schema subset supported by `flint-generate::Grammar`: objects with required and optional properties, arrays, strings, integers, numbers, booleans, null, `enum`, `const`, `anyOf` / `oneOf`, and `$defs` references. Output is guaranteed to be a valid prefix of the schema from the first token on.

## Caching

Assets are stored under `temp/<org>--<repo>/` and reused on later runs; a file is re-downloaded only when its size differs from the repo listing.

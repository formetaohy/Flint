# verify

A CLI example that loads a single local `.gguf` file and runs streaming chat inference against it. Unlike the other examples it does not talk to Hugging Face — point `--gguf` at any file on disk (e.g. one of the quantizations in `temp/Qwen3.5-0.8B/`) and the model family, chat template and tokenizer are detected from the GGUF metadata.

## Usage

```sh
cargo run -p thuban_examples --example verify -- --gguf temp/Qwen3.5-0.8B/Qwen3.5-0.8B-Q4_K_M.gguf
```

The file is hard-linked into a scratch directory so the loader sees a directory with exactly one checkpoint, and the scratch directory is removed after the run.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--gguf` | — | Path to the `.gguf` file to load (required) |
| `--prompt` | `What is 2+2? Answer briefly.` | User prompt |
| `--max-tokens` | 64 | Generation limit |
| `--ctx-size` | 4096 | Context length |

## Environment

- `VERIFY_IDS` — when set, every streamed piece is printed as `text[token_id]`, which is handy for comparing token sequences across quantizations of the same model.

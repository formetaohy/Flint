# server

A CLI example that downloads a Hugging Face model on demand and serves it behind OpenAI, Anthropic and Gemini compatible HTTP APIs.

## Usage

```sh
cargo run -p flint-examples --example server -- --model Qwen/Qwen3.5-0.6B-Instruct-2507 --port 8080
```

The server listens on `http://127.0.0.1:8080` and exposes four API surfaces over the same engine:

| Endpoint | Protocol |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/responses` | OpenAI Responses API |
| `POST /v1/messages`, `POST /v1/messages/count_tokens` | Anthropic Messages |
| `POST /v1beta/models/*:generateContent`, `POST /v1beta/models/*:streamGenerateContent` | Gemini `generateContent` |
| `GET /v1/models`, `GET /healthz` | Model listing / health check |

## Examples

```sh
# OpenAI chat completions (streaming)
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3.5-0.6B-Instruct-2507", "stream": true,
       "messages": [{"role": "user", "content": "Why is the sky blue?"}]}'

# OpenAI responses API
curl -N http://127.0.0.1:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5", "stream": true,
       "input": [{"type": "message", "role": "user",
                  "content": [{"type": "input_text", "text": "Hello"}]}]}'

# Anthropic messages (thinking blocks are streamed as thinking_delta events)
curl -N http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: local" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model": "qwen3.5", "stream": true, "max_tokens": 1024,
       "messages": [{"role": "user", "content": "Count to ten"}]}'

# Gemini generateContent
curl -N -X POST "http://127.0.0.1:8080/v1beta/models/qwen3.5:streamGenerateContent" \
  -H "Content-Type: application/json" \
  -d '{"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}'
```

## Thinking models

When the loaded model is a thinking model (e.g. Qwen3.5, Qwen3, DeepSeek-R1), the engine reasons inside `<think>...</think>` and the server splits the stream per protocol:

- OpenAI chat: reasoning arrives as `delta.reasoning_content` (and `usage.completion_tokens_details.reasoning_tokens` in the final response)
- Anthropic: reasoning arrives as `thinking` content blocks (`thinking_delta` events in streaming)
- Gemini: reasoning parts carry `"thought": true`
- Responses API: reasoning arrives as a `reasoning` output item (`response.reasoning_text.delta` events in streaming)

Thinking can be disabled per request: `enable_thinking: false` (chat), `thinking: {"type": "disabled"}` (Anthropic), `thinkingBudget: 0` (Gemini), `reasoning: {"effort": "none"}` (Responses). Note that requests using structured output (`response_format.json_schema`, Responses `text.format`) or tools run with grammar constraints from the first token, so thinking is disabled automatically for them.

## Caching

Assets are stored under `temp/<org>--<repo>/` and reused on later runs; a file is re-downloaded only when its size differs from the repo listing.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 8080 | Bind port |
| `--api-key` | none | When set, requests must carry it in `Authorization: Bearer`, `x-api-key` or `x-goog-api-key` |
| `--ctx-size` | 32768 | Context length |
| `--max-tokens` | 4096 | Default generation limit when the request omits it |
| `--seed` | 42 | Sampling seed |
| `--speculate` | off | Enable speculative decoding |

## Environment

- `HF_ENDPOINT` — override the Hub base URL (e.g. a mirror)

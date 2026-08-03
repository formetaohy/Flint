//! Deterministic toy checkpoints for the test suite. A toy is a
//! minimal-dimension model with fixed-seed random weights, written in its
//! native format (GGUF for the dense families, safetensors for Qwen3.5), so
//! the complete load -> forward -> generate path runs in tests with zero
//! external weight files.
//!
//! Structural invariants (chunked vs sequential prefill, speculative vs plain
//! decoding, snapshot/restore, tokenizer round-trips) are weight-independent:
//! they compare execution paths over identical inputs, not the quality of the
//! output. Toy checkpoints exist precisely to exercise those paths on CI.

use std::path::Path;

use super::tokenizer;
use flint_checkpoint::{GgufWriter, write_tensors};
use flint_error::{Error, Result};
use serde_json::json;

/// Which minimal checkpoint to materialize.
#[derive(Clone, Copy, Debug)]
pub enum ToySpec {
    /// Dense GQA, GGUF, untied head, no biases/QK-norm.
    Llama,
    /// Dense GQA with QKV projection biases (Qwen2-style).
    LlamaQkvBias,
    /// Dense GQA with per-head QK-norm (Qwen3-style).
    LlamaQkNorm,
    /// Gemma 3: sandwich norms, sliding window, always-on QK-norm, tied head.
    Gemma,
    /// Qwen3.5 hybrid Gated DeltaNet + MTP draft head, safetensors.
    Qwen35,
    /// Qwen3.5 with fewer linear key heads than value heads (repeat path).
    Qwen35Split,
    /// Qwen3.5 with untied embeddings (separate lm_head).
    Qwen35Untied,
}

impl ToySpec {
    /// Writes the checkpoint (plus tokenizer and config) into `dir`.
    pub fn write(self, dir: &Path) -> Result<()> {
        match self {
            ToySpec::Qwen35 => write_qwen35(dir, 16, 16, true),
            ToySpec::Qwen35Split => write_qwen35(dir, 8, 16, true),
            ToySpec::Qwen35Untied => write_qwen35(dir, 16, 16, false),
            other => write_dense(other, dir),
        }
    }
}

// ---------------------------------------------------------------- dims

/// Toy dimensions satisfy every kernel and config constraint: N % 16, K % 64,
/// head_dim in [64, 256], q heads a multiple of kv heads, vocab % 16.
const HIDDEN: u32 = 64;
const Q_HEADS: u32 = 4;
const KV_HEADS: u32 = 1;
const HEAD_DIM: u32 = 64;
const INTERMEDIATE: u32 = 128;
const LAYERS: u32 = 2;
const VOCAB: u32 = tokenizer::VOCAB as u32;

// ---------------------------------------------------------------- rng

/// Deterministic LCG, identical to the kernel test harness.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
    }
    fn fill(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// One f32 tensor in canonical (HF) naming, before format conversion.
struct Canon {
    key: String,
    shape: Vec<u32>,
    /// Storage role as the loader will re-pack it.
    role: Role,
}

#[derive(Clone, Copy)]
enum Role {
    /// Norms, biases, conv taps: stored f32.
    F32,
    /// Embedding table: packed bf16.
    Bf16,
    /// Projections: Q8_0 in GGUF, plain f32 in safetensors (re-quantized on
    /// upload either way).
    Proj,
}

fn canon(prefix: &str, key: &str) -> String {
    format!("{prefix}.{key}")
}

fn proj(key: String, shape: &[u32]) -> Canon {
    Canon {
        key,
        shape: shape.to_vec(),
        role: Role::Proj,
    }
}

fn f32w(key: String, shape: &[u32]) -> Canon {
    Canon {
        key,
        shape: shape.to_vec(),
        role: Role::F32,
    }
}

fn dense_layer_keys(l: u32, qkv_bias: bool, qk_norm: bool, sandwich: bool) -> Vec<Canon> {
    let p = format!("layers.{l}");
    let mut v = vec![
        f32w(canon(&p, "input_layernorm.weight"), &[HIDDEN]),
        proj(
            canon(&p, "self_attn.q_proj.weight"),
            &[Q_HEADS * HEAD_DIM, HIDDEN],
        ),
        proj(
            canon(&p, "self_attn.k_proj.weight"),
            &[KV_HEADS * HEAD_DIM, HIDDEN],
        ),
        proj(
            canon(&p, "self_attn.v_proj.weight"),
            &[KV_HEADS * HEAD_DIM, HIDDEN],
        ),
        proj(
            canon(&p, "self_attn.o_proj.weight"),
            &[HIDDEN, Q_HEADS * HEAD_DIM],
        ),
    ];
    if qkv_bias {
        v.push(f32w(
            canon(&p, "self_attn.q_proj.bias"),
            &[Q_HEADS * HEAD_DIM],
        ));
        v.push(f32w(
            canon(&p, "self_attn.k_proj.bias"),
            &[KV_HEADS * HEAD_DIM],
        ));
        v.push(f32w(
            canon(&p, "self_attn.v_proj.bias"),
            &[KV_HEADS * HEAD_DIM],
        ));
    }
    if qk_norm {
        v.push(f32w(canon(&p, "self_attn.q_norm.weight"), &[HEAD_DIM]));
        v.push(f32w(canon(&p, "self_attn.k_norm.weight"), &[HEAD_DIM]));
    }
    if sandwich {
        v.push(f32w(canon(&p, "post_attention_norm.weight"), &[HIDDEN]));
        v.push(f32w(canon(&p, "post_ffw_norm.weight"), &[HIDDEN]));
    }
    v.push(f32w(
        canon(&p, "post_attention_layernorm.weight"),
        &[HIDDEN],
    ));
    v.push(proj(
        canon(&p, "mlp.gate_proj.weight"),
        &[INTERMEDIATE, HIDDEN],
    ));
    v.push(proj(
        canon(&p, "mlp.up_proj.weight"),
        &[INTERMEDIATE, HIDDEN],
    ));
    v.push(proj(
        canon(&p, "mlp.down_proj.weight"),
        &[HIDDEN, INTERMEDIATE],
    ));
    v
}

// ---------------------------------------------------------------- dense (GGUF)

/// GGUF-native name for a dense canonical key; the exact inverse of
/// `flint_architectures::gguf_config::gguf_key`.
fn gguf_name(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("layers.") {
        let (idx, tail) = rest.split_once('.').unwrap();
        let (stem, suffix) = tail.rsplit_once('.').unwrap();
        let g = match stem {
            "input_layernorm" => "attn_norm",
            "self_attn.q_proj" => "attn_q",
            "self_attn.k_proj" => "attn_k",
            "self_attn.v_proj" => "attn_v",
            "self_attn.o_proj" => "attn_output",
            "self_attn.q_norm" => "attn_q_norm",
            "self_attn.k_norm" => "attn_k_norm",
            "post_attention_layernorm" => "ffn_norm",
            "mlp.gate_proj" => "ffn_gate",
            "mlp.up_proj" => "ffn_up",
            "mlp.down_proj" => "ffn_down",
            "post_attention_norm" => "post_attention_norm",
            "post_ffw_norm" => "post_ffw_norm",
            other => panic!("unmapped dense key {other}"),
        };
        return format!("blk.{idx}.{g}.{suffix}");
    }
    match key {
        "embed_tokens.weight" => "token_embd.weight".into(),
        "lm_head.weight" => "output.weight".into(),
        "norm.weight" => "output_norm.weight".into(),
        other => panic!("unmapped dense key {other}"),
    }
}

fn write_dense(spec: ToySpec, dir: &Path) -> Result<()> {
    let (arch, gemma) = match spec {
        ToySpec::Llama | ToySpec::LlamaQkvBias | ToySpec::LlamaQkNorm => ("llama", false),
        ToySpec::Gemma => ("gemma3", true),
        ToySpec::Qwen35 | ToySpec::Qwen35Split | ToySpec::Qwen35Untied => unreachable!(),
    };
    let (qkv_bias, qk_norm, sandwich, tied) = match spec {
        ToySpec::LlamaQkvBias => (true, false, false, false),
        ToySpec::LlamaQkNorm => (false, true, false, false),
        ToySpec::Gemma => (false, true, true, true),
        _ => (false, false, false, false),
    };

    let mut rng = Rng::new(0x5eed);
    let mut w = GgufWriter::new(32);
    w.kv_str("general.architecture", arch);
    w.kv_u32("general.alignment", 32);
    let key = |k: &str| format!("{arch}.{k}");
    w.kv_u32(&key("block_count"), LAYERS);
    w.kv_u32(&key("embedding_length"), HIDDEN);
    w.kv_u32(&key("attention.head_count"), Q_HEADS);
    w.kv_u32(&key("attention.head_count_kv"), KV_HEADS);
    w.kv_u32(&key("attention.key_length"), HEAD_DIM);
    w.kv_u32(&key("feed_forward_length"), INTERMEDIATE);
    w.kv_u32(&key("vocab_size"), VOCAB);
    if gemma {
        w.kv_u32(&key("attention.sliding_window"), 4);
        w.kv_u32(&key("attention.sliding_window_pattern"), 2);
    }

    let tokens = tokenizer::tokens();
    w.kv_str("tokenizer.ggml.model", "bpe");
    w.kv_str_array("tokenizer.ggml.tokens", &tokens);
    w.kv_str_array("tokenizer.ggml.merges", &tokenizer::merges());
    w.kv_u32_array("tokenizer.ggml.token_type", &tokenizer::token_types());
    w.kv_u32("tokenizer.ggml.eos_token_id", tokenizer::EOS_ID);

    let mut all = vec![
        Canon {
            key: "embed_tokens.weight".into(),
            shape: vec![VOCAB, HIDDEN],
            role: Role::Bf16,
        },
        Canon {
            key: "norm.weight".into(),
            shape: vec![HIDDEN],
            role: Role::F32,
        },
    ];
    if !tied {
        all.push(Canon {
            key: "lm_head.weight".into(),
            shape: vec![VOCAB, HIDDEN],
            role: Role::Proj,
        });
    }
    for l in 0..LAYERS {
        all.extend(dense_layer_keys(l, qkv_bias, qk_norm, sandwich));
    }
    for c in all {
        let data = rng.fill(c.shape.iter().map(|d| *d as usize).product());
        let name = gguf_name(&c.key);
        match c.role {
            Role::F32 => w.tensor_f32(&name, &c.shape, &data),
            Role::Bf16 => w.tensor_bf16(&name, &c.shape, &data),
            Role::Proj => w.tensor_q8_0(&name, &c.shape, &data),
        }
    }

    std::fs::create_dir_all(dir)
        .map_err(|e| Error::Model(format!("create {}: {e}", dir.display())))?;
    std::fs::write(dir.join("model.gguf"), w.finish())
        .map_err(|e| Error::Model(format!("write model.gguf: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------- Qwen3.5 (safetensors)

/// Materializes a Qwen3.5 toy with the given linear-attention key/value head
/// counts and embedding tying; the MTP draft head is always present.
fn write_qwen35(
    dir: &Path,
    lin_key_heads: u32,
    lin_val_heads: u32,
    tied: bool,
) -> Result<()> {
    let (lin_key, lin_val) = (8u32, 8u32);
    let key_dim = lin_key_heads * lin_key;
    let val_dim = lin_val_heads * lin_val;
    let conv_dim = key_dim * 2 + val_dim;

    // Full-attention layer weights under a prefix (target layers + MTP head).
    let full = |p: &str| -> Vec<Canon> {
        vec![
            f32w(format!("{p}.input_layernorm.weight"), &[HIDDEN]),
            proj(
                format!("{p}.self_attn.q_proj.weight"),
                &[Q_HEADS * HEAD_DIM * 2, HIDDEN],
            ),
            proj(
                format!("{p}.self_attn.k_proj.weight"),
                &[KV_HEADS * HEAD_DIM, HIDDEN],
            ),
            proj(
                format!("{p}.self_attn.v_proj.weight"),
                &[KV_HEADS * HEAD_DIM, HIDDEN],
            ),
            proj(
                format!("{p}.self_attn.o_proj.weight"),
                &[HIDDEN, Q_HEADS * HEAD_DIM],
            ),
            f32w(format!("{p}.self_attn.q_norm.weight"), &[HEAD_DIM]),
            f32w(format!("{p}.self_attn.k_norm.weight"), &[HEAD_DIM]),
            f32w(format!("{p}.post_attention_layernorm.weight"), &[HIDDEN]),
            proj(format!("{p}.mlp.gate_proj.weight"), &[INTERMEDIATE, HIDDEN]),
            proj(format!("{p}.mlp.up_proj.weight"), &[INTERMEDIATE, HIDDEN]),
            proj(format!("{p}.mlp.down_proj.weight"), &[HIDDEN, INTERMEDIATE]),
        ]
    };
    // Gated DeltaNet layer weights under its prefix.
    let linear = |p: &str| -> Vec<Canon> {
        vec![
            f32w(format!("{p}.input_layernorm.weight"), &[HIDDEN]),
            proj(
                format!("{p}.linear_attn.in_proj_qkv.weight"),
                &[conv_dim, HIDDEN],
            ),
            proj(
                format!("{p}.linear_attn.in_proj_z.weight"),
                &[val_dim, HIDDEN],
            ),
            proj(
                format!("{p}.linear_attn.in_proj_b.weight"),
                &[lin_val_heads, HIDDEN],
            ),
            proj(
                format!("{p}.linear_attn.in_proj_a.weight"),
                &[lin_val_heads, HIDDEN],
            ),
            f32w(format!("{p}.linear_attn.conv1d.weight"), &[conv_dim, 4]),
            f32w(format!("{p}.linear_attn.A_log"), &[lin_val_heads]),
            f32w(format!("{p}.linear_attn.dt_bias"), &[lin_val_heads]),
            f32w(format!("{p}.linear_attn.norm.weight"), &[val_dim]),
            proj(
                format!("{p}.linear_attn.out_proj.weight"),
                &[HIDDEN, val_dim],
            ),
            f32w(format!("{p}.post_attention_layernorm.weight"), &[HIDDEN]),
            proj(format!("{p}.mlp.gate_proj.weight"), &[INTERMEDIATE, HIDDEN]),
            proj(format!("{p}.mlp.up_proj.weight"), &[INTERMEDIATE, HIDDEN]),
            proj(format!("{p}.mlp.down_proj.weight"), &[HIDDEN, INTERMEDIATE]),
        ]
    };

    let mut all = vec![
        Canon {
            key: "model.language_model.embed_tokens.weight".into(),
            shape: vec![VOCAB, HIDDEN],
            role: Role::Bf16,
        },
        Canon {
            key: "model.language_model.norm.weight".into(),
            shape: vec![HIDDEN],
            role: Role::F32,
        },
    ];
    if !tied {
        all.push(Canon {
            key: "lm_head.weight".into(),
            shape: vec![VOCAB, HIDDEN],
            role: Role::Proj,
        });
    }
    let lang = |p: &str| format!("model.language_model.{p}");
    all.extend(linear(&lang("layers.0")));
    all.extend(full(&lang("layers.1")));
    all.extend(vec![
        f32w("mtp.pre_fc_norm_embedding.weight".into(), &[HIDDEN]),
        f32w("mtp.pre_fc_norm_hidden.weight".into(), &[HIDDEN]),
        proj("mtp.fc.weight".into(), &[HIDDEN, 2 * HIDDEN]),
        f32w("mtp.norm.weight".into(), &[HIDDEN]),
    ]);
    all.extend(full("mtp.layers.0"));

    let mut rng = Rng::new(0xdead);
    let mut files: Vec<(String, Vec<u32>, Vec<u8>, bool)> = Vec::new();
    for c in &all {
        let data = rng.fill(c.shape.iter().map(|d| *d as usize).product());
        let bytes = match c.role {
            Role::F32 | Role::Proj => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Role::Bf16 => data
                .iter()
                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                .collect(),
        };
        files.push((
            c.key.clone(),
            c.shape.to_vec(),
            bytes,
            matches!(c.role, Role::Bf16),
        ));
    }

    std::fs::create_dir_all(dir)
        .map_err(|e| Error::Model(format!("create {}: {e}", dir.display())))?;
    write_tensors(&dir.join("model.safetensors"), &files)?;
    std::fs::write(dir.join("config.json"), qwen35_config(lin_key_heads, lin_val_heads, tied).to_string())
        .map_err(|e| Error::Model(format!("write config.json: {e}")))?;
    std::fs::write(dir.join("tokenizer.json"), tokenizer::tokenizer_json()?)
        .map_err(|e| Error::Model(format!("write tokenizer.json: {e}")))?;
    Ok(())
}

fn qwen35_config(lin_key_heads: u32, lin_val_heads: u32, tied: bool) -> serde_json::Value {
    json!({
        "model_type": "qwen3_5",
        "tie_word_embeddings": tied,
        "text_config": {
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": Q_HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "vocab_size": VOCAB,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": lin_key_heads,
            "linear_num_value_heads": lin_val_heads,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "mtp_num_hidden_layers": 1,
            "rope_parameters": { "rope_theta": 10000.0, "partial_rotary_factor": 0.5 },
            "eos_token_id": [tokenizer::EOS_ID],
        }
    })
}

use std::path::Path;

use super::tokenizer;
use flint_checkpoint::{GgufWriter, write_tensors};
use flint_error::{Error, Result};
use serde_json::json;

#[derive(Clone, Copy, Debug)]
pub enum ToySpec {

    Llama,

    LlamaQkvBias,

    LlamaQkNorm,

    Gemma,

    Qwen35,

    Qwen35Split,

    Qwen35Untied,

    Phi,

    PhiLayerNorm,

    PhiMoe,

    Gemma4,
}

impl ToySpec {

    pub fn write(self, dir: &Path) -> Result<()> {
        match self {
            ToySpec::Qwen35 => write_qwen35(dir, 16, 16, true),
            ToySpec::Qwen35Split => write_qwen35(dir, 8, 16, true),
            ToySpec::Qwen35Untied => write_qwen35(dir, 16, 16, false),
            ToySpec::Phi => write_phi(dir),
            ToySpec::PhiLayerNorm => write_phi_layernorm(dir),
            ToySpec::PhiMoe => write_phimoe(dir),
            ToySpec::Gemma4 => write_gemma4(dir),
            other => write_dense(other, dir),
        }
    }
}

const HIDDEN: u32 = 64;
const Q_HEADS: u32 = 4;
const KV_HEADS: u32 = 1;
const HEAD_DIM: u32 = 64;
const INTERMEDIATE: u32 = 128;
const LAYERS: u32 = 2;
const VOCAB: u32 = tokenizer::VOCAB as u32;

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

struct Canon {
    key: String,
    shape: Vec<u32>,

    role: Role,
}

#[derive(Clone, Copy)]
enum Role {

    F32,

    Bf16,

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

fn gguf_name(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("layers.") {
        let (idx, tail) = rest.split_once('.').unwrap();

        let (stem, suffix) = match tail.rsplit_once('.') {
            Some((s, f)) => (s, format!(".{f}")),
            None => (tail, ".weight".to_string()),
        };
        let g = match stem {
            "per_layer_input_gate" => "inp_gate",
            "per_layer_projection" => "proj",
            "post_per_layer_input_norm" => "post_norm",
            "layer_scalar" => "layer_output_scale",
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
        return format!("blk.{idx}.{g}{suffix}");
    }
    match key {
        "embed_tokens.weight" => "token_embd.weight".into(),
        "lm_head.weight" => "output.weight".into(),
        "norm.weight" => "output_norm.weight".into(),
        "embed_tokens_per_layer.weight" => "per_layer_token_embd.weight".into(),
        "per_layer_model_projection.weight" => "per_layer_model_proj.weight".into(),
        "per_layer_projection_norm.weight" => "per_layer_proj_norm.weight".into(),
        "rope_freqs.weight" => "rope_freqs.weight".into(),
        other => panic!("unmapped dense key {other}"),
    }
}

fn write_dense(spec: ToySpec, dir: &Path) -> Result<()> {
    let (arch, gemma) = match spec {
        ToySpec::Llama | ToySpec::LlamaQkvBias | ToySpec::LlamaQkNorm => ("llama", false),
        ToySpec::Gemma => ("gemma3", true),
        _ => unreachable!(),
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

fn write_qwen35(dir: &Path, lin_key_heads: u32, lin_val_heads: u32, tied: bool) -> Result<()> {
    let (lin_key, lin_val) = (8u32, 8u32);
    let key_dim = lin_key_heads * lin_key;
    let val_dim = lin_val_heads * lin_val;
    let conv_dim = key_dim * 2 + val_dim;

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
    std::fs::write(
        dir.join("config.json"),
        qwen35_config(lin_key_heads, lin_val_heads, tied).to_string(),
    )
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

fn write_phi(dir: &Path) -> Result<()> {
    let mut rng = Rng::new(0xf1a0);
    let mut w = GgufWriter::new(32);
    w.kv_str("general.architecture", "phi3");
    w.kv_u32("general.alignment", 32);
    let key = |k: &str| format!("phi3.{k}");
    w.kv_u32(&key("block_count"), LAYERS);
    w.kv_u32(&key("embedding_length"), HIDDEN);
    w.kv_u32(&key("attention.head_count"), Q_HEADS);
    w.kv_u32(&key("attention.head_count_kv"), KV_HEADS);
    w.kv_u32(&key("attention.key_length"), HEAD_DIM);
    w.kv_u32(&key("feed_forward_length"), INTERMEDIATE);
    w.kv_u32(&key("vocab_size"), VOCAB);
    w.kv_u32(&key("rope.dimension_count"), HEAD_DIM / 2);
    w.kv_f32(&key("rope.freq_base"), 10000.0);
    w.kv_f32(&key("attention.layer_norm_rms_epsilon"), 1e-5);
    write_gguf_tokenizer(&mut w);
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
    for l in 0..LAYERS {
        all.extend(dense_layer_keys(l, false, false, false));
    }
    write_gguf_tensors(&mut w, &mut rng, &all, dir)
}

fn write_phi_layernorm(dir: &Path) -> Result<()> {
    let mut rng = Rng::new(0x1a10);
    let mut all = vec![
        Canon {
            key: "model.embed_tokens.weight".into(),
            shape: vec![VOCAB, HIDDEN],
            role: Role::Bf16,
        },
        Canon {
            key: "model.norm.weight".into(),
            shape: vec![HIDDEN],
            role: Role::F32,
        },
        Canon {
            key: "model.norm.bias".into(),
            shape: vec![HIDDEN],
            role: Role::F32,
        },
        Canon {
            key: "lm_head.weight".into(),
            shape: vec![VOCAB, HIDDEN],
            role: Role::Proj,
        },
    ];
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        all.push(f32w(format!("{p}.input_layernorm.weight"), &[HIDDEN]));
        all.push(f32w(format!("{p}.input_layernorm.bias"), &[HIDDEN]));
        all.push(proj(
            format!("{p}.self_attn.q_proj.weight"),
            &[Q_HEADS * HEAD_DIM, HIDDEN],
        ));
        all.push(proj(
            format!("{p}.self_attn.k_proj.weight"),
            &[KV_HEADS * HEAD_DIM, HIDDEN],
        ));
        all.push(proj(
            format!("{p}.self_attn.v_proj.weight"),
            &[KV_HEADS * HEAD_DIM, HIDDEN],
        ));
        all.push(proj(
            format!("{p}.self_attn.o_proj.weight"),
            &[HIDDEN, Q_HEADS * HEAD_DIM],
        ));
        all.push(f32w(
            format!("{p}.post_attention_layernorm.weight"),
            &[HIDDEN],
        ));
        all.push(f32w(
            format!("{p}.post_attention_layernorm.bias"),
            &[HIDDEN],
        ));
        all.push(proj(
            format!("{p}.mlp.gate_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
        ));
        all.push(proj(
            format!("{p}.mlp.up_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
        ));
        all.push(proj(
            format!("{p}.mlp.down_proj.weight"),
            &[HIDDEN, INTERMEDIATE],
        ));
    }

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
    let cfg = json!({
        "model_type": "phi3",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rope_theta": 10000.0,
        "eos_token_id": tokenizer::EOS_ID,
        "tie_word_embeddings": false,
        "partial_rotary_factor": 0.5,
        "layer_norm_epsilon": 1e-5,
    });
    std::fs::write(dir.join("config.json"), cfg.to_string())
        .map_err(|e| Error::Model(format!("write config.json: {e}")))?;
    std::fs::write(dir.join("tokenizer.json"), tokenizer::tokenizer_json()?)
        .map_err(|e| Error::Model(format!("write tokenizer.json: {e}")))?;
    Ok(())
}

fn write_phimoe(dir: &Path) -> Result<()> {
    const EXPERTS: u32 = 16;
    let mut rng = Rng::new(0x100e);
    let mut all = vec![
        Canon {
            key: "model.embed_tokens.weight".into(),
            shape: vec![VOCAB, HIDDEN],
            role: Role::Bf16,
        },
        Canon {
            key: "model.norm.weight".into(),
            shape: vec![HIDDEN],
            role: Role::F32,
        },
        Canon {
            key: "model.norm.bias".into(),
            shape: vec![HIDDEN],
            role: Role::F32,
        },
        Canon {
            key: "lm_head.weight".into(),
            shape: vec![VOCAB, HIDDEN],
            role: Role::Proj,
        },
        Canon {
            key: "lm_head.bias".into(),
            shape: vec![VOCAB],
            role: Role::F32,
        },
    ];
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        all.push(f32w(format!("{p}.input_layernorm.weight"), &[HIDDEN]));
        all.push(f32w(format!("{p}.input_layernorm.bias"), &[HIDDEN]));
        all.push(f32w(
            format!("{p}.post_attention_layernorm.weight"),
            &[HIDDEN],
        ));
        all.push(f32w(
            format!("{p}.post_attention_layernorm.bias"),
            &[HIDDEN],
        ));
        all.push(proj(
            format!("{p}.self_attn.q_proj.weight"),
            &[Q_HEADS * HEAD_DIM, HIDDEN],
        ));
        all.push(proj(
            format!("{p}.self_attn.k_proj.weight"),
            &[KV_HEADS * HEAD_DIM, HIDDEN],
        ));
        all.push(proj(
            format!("{p}.self_attn.v_proj.weight"),
            &[KV_HEADS * HEAD_DIM, HIDDEN],
        ));
        all.push(proj(
            format!("{p}.self_attn.o_proj.weight"),
            &[HIDDEN, Q_HEADS * HEAD_DIM],
        ));
        all.push(f32w(
            format!("{p}.self_attn.q_proj.bias"),
            &[Q_HEADS * HEAD_DIM],
        ));
        all.push(f32w(
            format!("{p}.self_attn.k_proj.bias"),
            &[KV_HEADS * HEAD_DIM],
        ));
        all.push(f32w(
            format!("{p}.self_attn.v_proj.bias"),
            &[KV_HEADS * HEAD_DIM],
        ));

        all.push(Canon {
            key: format!("{p}.mlp.gate_up_proj"),
            shape: vec![EXPERTS, 2 * INTERMEDIATE, HIDDEN],
            role: Role::Proj,
        });
        all.push(Canon {
            key: format!("{p}.mlp.down_proj"),
            shape: vec![EXPERTS, HIDDEN, INTERMEDIATE],
            role: Role::Proj,
        });
        all.push(proj(format!("{p}.mlp.router.weight"), &[EXPERTS, HIDDEN]));
    }

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
    let cfg = json!({
        "model_type": "phimoe",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rope_theta": 10000.0,
        "eos_token_id": tokenizer::EOS_ID,
        "tie_word_embeddings": false,
        "num_local_experts": EXPERTS,
        "num_experts_per_tok": 2,
        "sliding_window": 4,
        "rms_norm_eps": 1e-5,
        "lm_head_bias": true,
        "attention_bias": true,
    });
    std::fs::write(dir.join("config.json"), cfg.to_string())
        .map_err(|e| Error::Model(format!("write config.json: {e}")))?;
    std::fs::write(dir.join("tokenizer.json"), tokenizer::tokenizer_json()?)
        .map_err(|e| Error::Model(format!("write tokenizer.json: {e}")))?;
    Ok(())
}

fn write_gemma4(dir: &Path) -> Result<()> {
    const PLE: u32 = 64;
    const FULL_HD: u32 = 128;
    let mut rng = Rng::new(0x6e4a);
    let mut w = GgufWriter::new(32);
    w.kv_str("general.architecture", "gemma4");
    w.kv_u32("general.alignment", 32);
    let key = |k: &str| format!("gemma4.{k}");
    w.kv_u32(&key("block_count"), 4);
    w.kv_u32(&key("embedding_length"), HIDDEN);
    w.kv_u32(&key("attention.head_count"), Q_HEADS);
    w.kv_u32(&key("attention.head_count_kv"), KV_HEADS);
    w.kv_u32(&key("attention.key_length"), FULL_HD);
    w.kv_u32(&key("attention.key_length_swa"), HEAD_DIM);
    w.kv_u32(&key("attention.value_length"), FULL_HD);
    w.kv_u32(&key("attention.value_length_swa"), HEAD_DIM);
    w.kv_u32(&key("attention.sliding_window"), 4);
    w.kv_u32_array(&key("attention.sliding_window_pattern"), &[1, 0, 1, 0]);
    w.kv_u32(&key("attention.shared_kv_layers"), 2);
    w.kv_u32_array(&key("feed_forward_length"), &[128, 128, 256, 256]);
    w.kv_u32(&key("vocab_size"), VOCAB);
    w.kv_f32(&key("attention.layer_norm_rms_epsilon"), 1e-6);
    w.kv_f32(&key("final_logit_softcapping"), 30.0);
    w.kv_u32(&key("embedding_length_per_layer_input"), PLE);
    w.kv_f32(&key("rope.freq_base"), 1000000.0);
    w.kv_f32(&key("rope.freq_base_swa"), 10000.0);
    w.kv_u32(&key("rope.dimension_count"), FULL_HD);
    w.kv_u32(&key("rope.dimension_count_swa"), HEAD_DIM);
    write_gguf_tokenizer(&mut w);

    let mut all = vec![
        Canon {
            key: "embed_tokens.weight".into(),
            shape: vec![VOCAB, HIDDEN],
            role: Role::Bf16,
        },
        Canon {
            key: "embed_tokens_per_layer.weight".into(),
            shape: vec![VOCAB, 4 * PLE],
            role: Role::Bf16,
        },
        Canon {
            key: "per_layer_model_projection.weight".into(),
            shape: vec![4 * PLE, HIDDEN],
            role: Role::Proj,
        },
        Canon {
            key: "per_layer_projection_norm.weight".into(),
            shape: vec![PLE],
            role: Role::F32,
        },
        Canon {
            key: "norm.weight".into(),
            shape: vec![HIDDEN],
            role: Role::F32,
        },
        Canon {
            key: "rope_freqs.weight".into(),
            shape: vec![FULL_HD / 2],
            role: Role::F32,
        },
    ];
    for l in 0..4u32 {
        let hd = if l % 2 == 0 { HEAD_DIM } else { FULL_HD };
        let ffl = if l >= 2 {
            2 * INTERMEDIATE
        } else {
            INTERMEDIATE
        };
        let has_kv = l < 2;
        let p = format!("layers.{l}");
        all.push(f32w(format!("{p}.input_layernorm.weight"), &[HIDDEN]));
        all.push(proj(
            format!("{p}.self_attn.q_proj.weight"),
            &[Q_HEADS * hd, HIDDEN],
        ));
        if has_kv {
            all.push(proj(
                format!("{p}.self_attn.k_proj.weight"),
                &[KV_HEADS * hd, HIDDEN],
            ));
            all.push(proj(
                format!("{p}.self_attn.v_proj.weight"),
                &[KV_HEADS * hd, HIDDEN],
            ));
        }
        all.push(proj(
            format!("{p}.self_attn.o_proj.weight"),
            &[HIDDEN, Q_HEADS * hd],
        ));
        all.push(f32w(format!("{p}.self_attn.q_norm.weight"), &[hd]));
        if has_kv {
            all.push(f32w(format!("{p}.self_attn.k_norm.weight"), &[hd]));
        }
        all.push(f32w(
            format!("{p}.post_attention_layernorm.weight"),
            &[HIDDEN],
        ));
        all.push(proj(format!("{p}.mlp.gate_proj.weight"), &[ffl, HIDDEN]));
        all.push(proj(format!("{p}.mlp.up_proj.weight"), &[ffl, HIDDEN]));
        all.push(proj(format!("{p}.mlp.down_proj.weight"), &[HIDDEN, ffl]));
        all.push(proj(
            format!("{p}.per_layer_input_gate.weight"),
            &[PLE, HIDDEN],
        ));
        all.push(proj(
            format!("{p}.per_layer_projection.weight"),
            &[HIDDEN, PLE],
        ));
        all.push(f32w(
            format!("{p}.post_per_layer_input_norm.weight"),
            &[HIDDEN],
        ));
        all.push(f32w(format!("{p}.layer_scalar"), &[1]));
    }
    write_gguf_tensors(&mut w, &mut rng, &all, dir)
}

fn write_gguf_tokenizer(w: &mut GgufWriter) {
    let tokens = tokenizer::tokens();
    w.kv_str("tokenizer.ggml.model", "bpe");
    w.kv_str_array("tokenizer.ggml.tokens", &tokens);
    w.kv_str_array("tokenizer.ggml.merges", &tokenizer::merges());
    w.kv_u32_array("tokenizer.ggml.token_type", &tokenizer::token_types());
    w.kv_u32("tokenizer.ggml.eos_token_id", tokenizer::EOS_ID);
}

fn write_gguf_tensors(w: &mut GgufWriter, rng: &mut Rng, all: &[Canon], dir: &Path) -> Result<()> {
    for c in all {
        let data = rng.fill(c.shape.iter().map(|d| *d as usize).product());
        match c.role {
            Role::F32 => w.tensor_f32(&gguf_name(&c.key), &c.shape, &data),
            Role::Bf16 => w.tensor_bf16(&gguf_name(&c.key), &c.shape, &data),
            Role::Proj => {
                if c.shape.len() == 3 {
                    let (e, n, k) = (c.shape[0], c.shape[1], c.shape[2]);
                    let mut flat = Vec::with_capacity(data.len());
                    for i in 0..e {
                        for r in 0..n {
                            let base = (i * n + r) * k;
                            flat.extend_from_slice(&data[base as usize..(base + k) as usize]);
                        }
                    }
                    w.tensor_q8_0(&gguf_name(&c.key), &c.shape, &flat)
                } else {
                    w.tensor_q8_0(&gguf_name(&c.key), &c.shape, &data)
                }
            }
        }
    }
    let w = std::mem::replace(w, GgufWriter::new(32));
    std::fs::create_dir_all(dir)
        .map_err(|e| Error::Model(format!("create {}: {e}", dir.display())))?;
    std::fs::write(dir.join("tokenizer.json"), tokenizer::tokenizer_json()?)
        .map_err(|e| Error::Model(format!("write tokenizer.json: {e}")))?;
    std::fs::write(dir.join("model.gguf"), w.finish())
        .map_err(|e| Error::Model(format!("write model.gguf: {e}")))?;
    Ok(())
}

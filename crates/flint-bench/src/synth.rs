use std::collections::HashMap;

use flint_checkpoint::{Checkpoint, CheckpointKind, Metadata, RawTensor, TensorData};
use flint_error::Result;

#[derive(Clone, Copy, Debug)]
pub struct BenchSpec {
    pub hidden: u32,
    pub intermediate: u32,
    pub layers: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub vocab: u32,
}

impl BenchSpec {

    pub fn weight_bytes(&self) -> u64 {
        let h = self.hidden as u64;
        let i = self.intermediate as u64;
        let l = self.layers as u64;
        let per_layer = (4 * h * h) + (3 * i * h) + (h + h) * 4;
        let proj_bytes = per_layer * l * 33 / 32; 
        let embed = self.vocab as u64 * h * 2; 
        let head = self.vocab as u64 * h * 33 / 32; 
        proj_bytes + embed + head + h * 4
    }
}

fn tensors(s: &BenchSpec) -> Vec<(String, Vec<u32>)> {
    let mut v = vec![
        ("model.embed_tokens.weight".into(), vec![s.vocab, s.hidden]),
        ("lm_head.weight".into(), vec![s.vocab, s.hidden]),
        ("model.norm.weight".into(), vec![s.hidden]),
    ];
    for l in 0..s.layers {
        let p = format!("model.layers.{l}");
        v.push((format!("{p}.input_layernorm.weight"), vec![s.hidden]));
        v.push((
            format!("{p}.post_attention_layernorm.weight"),
            vec![s.hidden],
        ));
        let qk = s.q_heads * s.head_dim;
        let vk = s.kv_heads * s.head_dim;
        v.push((format!("{p}.self_attn.q_proj.weight"), vec![qk, s.hidden]));
        v.push((format!("{p}.self_attn.k_proj.weight"), vec![vk, s.hidden]));
        v.push((format!("{p}.self_attn.v_proj.weight"), vec![vk, s.hidden]));
        v.push((format!("{p}.self_attn.o_proj.weight"), vec![s.hidden, qk]));
        v.push((
            format!("{p}.mlp.gate_proj.weight"),
            vec![s.intermediate, s.hidden],
        ));
        v.push((
            format!("{p}.mlp.up_proj.weight"),
            vec![s.intermediate, s.hidden],
        ));
        v.push((
            format!("{p}.mlp.down_proj.weight"),
            vec![s.hidden, s.intermediate],
        ));
    }
    v
}

fn seed_of(name: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn fill(seed: u64, gain: f32, out: &mut [f32]) {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0
    };
    for v in out.iter_mut() {
        *v = next() * gain;
    }
}

pub struct SynthCheckpoint {
    hidden: u32,

    index: HashMap<String, (Vec<u32>, u64)>,
}

impl SynthCheckpoint {
    pub fn new(spec: BenchSpec) -> Self {
        let index = tensors(&spec)
            .into_iter()
            .map(|(name, shape)| {
                let seed = seed_of(&name);
                (name, (shape, seed))
            })
            .collect();
        Self {
            hidden: spec.hidden,
            index,
        }
    }
}

impl Checkpoint for SynthCheckpoint {
    fn names(&self) -> Vec<String> {
        self.index.keys().cloned().collect()
    }

    fn read(&self, name: &str) -> Result<RawTensor> {
        let (shape, seed) = self
            .index
            .get(name)
            .ok_or_else(|| flint_error::Error::Model(format!("unknown tensor {name}")))?;
        let n = shape.iter().map(|d| *d as usize).product();
        let mut data = vec![0f32; n];

        let gain = if name.ends_with("_proj.weight") || name.ends_with("lm_head.weight") {
            1.0 / (self.hidden as f32).sqrt()
        } else {
            1.0
        };
        fill(*seed, gain, &mut data);
        Ok(RawTensor {
            shape: shape.clone(),
            data: TensorData::F32(data),
        })
    }

    fn metadata(&self) -> &Metadata {
        static EMPTY: std::sync::LazyLock<Metadata> = std::sync::LazyLock::new(Metadata::default);
        &EMPTY
    }

    fn config_json(&self) -> Result<Option<serde_json::Value>> {
        Ok(None)
    }

    fn kind(&self) -> CheckpointKind {
        CheckpointKind::Safetensors
    }
}

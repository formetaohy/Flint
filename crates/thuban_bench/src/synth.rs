use std::collections::HashMap;

use thuban_checkpoint::{Checkpoint, Metadata, RawTensor, TensorData};
use thuban_error::Result;

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

    pub fn config_json(&self) -> serde_json::Value {
        serde_json::json!({
            "hidden_size": self.hidden,
            "intermediate_size": self.intermediate,
            "num_hidden_layers": self.layers,
            "num_attention_heads": self.q_heads,
            "num_key_value_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab,
            "rope_theta": 500000.0,
            "eos_token_id": [0],
            "tie_word_embeddings": false,
        })
    }
}

fn tensors(s: &BenchSpec) -> Vec<(String, Vec<u32>)> {
    let mut v = vec![
        ("token_embd.weight".into(), vec![s.vocab, s.hidden]),
        ("output.weight".into(), vec![s.vocab, s.hidden]),
        ("output_norm.weight".into(), vec![s.hidden]),
    ];
    for l in 0..s.layers {
        let p = format!("blk.{l}");
        v.push((format!("{p}.attn_norm.weight"), vec![s.hidden]));
        v.push((format!("{p}.ffn_norm.weight"), vec![s.hidden]));
        let qk = s.q_heads * s.head_dim;
        let vk = s.kv_heads * s.head_dim;
        v.push((format!("{p}.attn_q.weight"), vec![qk, s.hidden]));
        v.push((format!("{p}.attn_k.weight"), vec![vk, s.hidden]));
        v.push((format!("{p}.attn_v.weight"), vec![vk, s.hidden]));
        v.push((format!("{p}.attn_output.weight"), vec![s.hidden, qk]));
        v.push((
            format!("{p}.ffn_gate.weight"),
            vec![s.intermediate, s.hidden],
        ));
        v.push((format!("{p}.ffn_up.weight"), vec![s.intermediate, s.hidden]));
        v.push((
            format!("{p}.ffn_down.weight"),
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
    meta: Metadata,
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
        let mut kv = HashMap::new();
        kv.insert(
            "general.architecture".to_string(),
            thuban_checkpoint::MetaVal::Str("llama".into()),
        );
        Self {
            hidden: spec.hidden,
            index,
            meta: Metadata::new(kv),
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
            .ok_or_else(|| thuban_error::Error::Model(format!("unknown tensor {name}")))?;
        let n = shape.iter().map(|d| *d as usize).product();
        let mut data = vec![0f32; n];

        let gain = if name.ends_with(".weight")
            && !name.contains("norm")
            && name != "token_embd.weight"
        {
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

    fn metadata(&self) -> Result<&Metadata> {
        Ok(&self.meta)
    }
}

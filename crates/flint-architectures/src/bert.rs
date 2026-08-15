use flint_backend::{Backend, Binding, Commands};
use flint_checkpoint::Checkpoint;
use flint_error::{Error, Result};
use flint_model::loader::{self, Plan, Role, WeightSet};
use flint_model::ops::{self, Act, NormMode, NormSpec};
use flint_model::pool::KvPool;
use flint_model::step;
use flint_model::TextEmbedder;
use flint_tensor::{Tensor, Weight};
use serde_json::Value;

const MAX_TOKENS: u32 = 512;

const PLAN: Plan = Plan { key: hf_key, role };

fn hf_key(name: &str) -> Option<String> {
    if name.starts_with("embeddings.") {
        return Some(name.to_string());
    }
    let rest = name.strip_prefix("encoder.layer.")?;
    let (idx, tail) = rest.split_once('.')?;
    let key = match tail {
        "attention.self.query.weight" => format!("layers.{idx}.self_attn.q.weight"),
        "attention.self.query.bias" => format!("layers.{idx}.self_attn.q.bias"),
        "attention.self.key.weight" => format!("layers.{idx}.self_attn.k.weight"),
        "attention.self.key.bias" => format!("layers.{idx}.self_attn.k.bias"),
        "attention.self.value.weight" => format!("layers.{idx}.self_attn.v.weight"),
        "attention.self.value.bias" => format!("layers.{idx}.self_attn.v.bias"),
        "attention.output.dense.weight" => format!("layers.{idx}.self_attn.o.weight"),
        "attention.output.dense.bias" => format!("layers.{idx}.self_attn.o.bias"),
        "attention.output.LayerNorm.weight" => format!("layers.{idx}.attn_ln.weight"),
        "attention.output.LayerNorm.bias" => format!("layers.{idx}.attn_ln.bias"),
        "intermediate.dense.weight" => format!("layers.{idx}.ffn.int.weight"),
        "intermediate.dense.bias" => format!("layers.{idx}.ffn.int.bias"),
        "output.dense.weight" => format!("layers.{idx}.ffn.out.weight"),
        "output.dense.bias" => format!("layers.{idx}.ffn.out.bias"),
        "output.LayerNorm.weight" => format!("layers.{idx}.ffn_ln.weight"),
        "output.LayerNorm.bias" => format!("layers.{idx}.ffn_ln.bias"),
        _ => return None,
    };
    Some(key)
}

fn role(key: &str) -> Role {
    if key.contains("_ln") || key.contains("LayerNorm") || key.ends_with(".bias") {
        Role::F32
    } else if key.contains("embeddings") {
        Role::Bf16
    } else {
        Role::I8
    }
}

struct Config {
    hidden: u32,
    intermediate: u32,
    layers: u32,
    heads: u32,
    max_pos: u32,
    head_dim: u32,
}

impl Config {
    fn parse(v: &Value) -> Result<Self> {
        let u = |k: &str| {
            v.get(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| Error::Config(format!("missing {k}")))
                .map(|x| x as u32)
        };
        let heads = u("num_attention_heads")?;
        let hidden = u("hidden_size")?;
        let cfg = Self {
            hidden,
            intermediate: u("intermediate_size")?,
            layers: u("num_hidden_layers")?,
            heads,
            max_pos: u("max_position_embeddings")?,
            head_dim: hidden / heads,
        };
        if !hidden.is_multiple_of(heads) {
            return Err(Error::Config("hidden not divisible by heads".into()));
        }
        if cfg.max_pos < MAX_TOKENS {
            return Err(Error::Config(format!(
                "max_position_embeddings {} below supported maximum {MAX_TOKENS}",
                cfg.max_pos
            )));
        }
        Ok(cfg)
    }
}

struct LayerW {
    attn_ln: Tensor,
    attn_ln_bias: Tensor,
    q: Weight,
    q_bias: Tensor,
    k: Weight,
    k_bias: Tensor,
    v: Weight,
    v_bias: Tensor,
    o: Weight,
    o_bias: Tensor,
    ffn_ln: Tensor,
    ffn_ln_bias: Tensor,
    int: Weight,
    int_bias: Tensor,
    out: Weight,
    out_bias: Tensor,
}

fn take_layer(w: &mut WeightSet, idx: usize) -> Result<LayerW> {
    let k = |n: &str| format!("layers.{idx}.{n}");
    Ok(LayerW {
        attn_ln: w.take_tensor(&k("attn_ln.weight"))?,
        attn_ln_bias: w.take_tensor(&k("attn_ln.bias"))?,
        q: w.take(&k("self_attn.q.weight"))?,
        q_bias: w.take_tensor(&k("self_attn.q.bias"))?,
        k: w.take(&k("self_attn.k.weight"))?,
        k_bias: w.take_tensor(&k("self_attn.k.bias"))?,
        v: w.take(&k("self_attn.v.weight"))?,
        v_bias: w.take_tensor(&k("self_attn.v.bias"))?,
        o: w.take(&k("self_attn.o.weight"))?,
        o_bias: w.take_tensor(&k("self_attn.o.bias"))?,
        ffn_ln: w.take_tensor(&k("ffn_ln.weight"))?,
        ffn_ln_bias: w.take_tensor(&k("ffn_ln.bias"))?,
        int: w.take(&k("ffn.int.weight"))?,
        int_bias: w.take_tensor(&k("ffn.int.bias"))?,
        out: w.take(&k("ffn.out.weight"))?,
        out_bias: w.take_tensor(&k("ffn.out.bias"))?,
    })
}

struct Scratch {
    ids: Tensor,
    pos_ids: Tensor,
    zero_ids: Tensor,
    meta: Tensor,
    hidden: Tensor,
    normed: Tensor,
    q_out: Tensor,
    k_out: Tensor,
    v_out: Tensor,
    attn_out: Tensor,
    int_out: Tensor,
    act_out: Tensor,
    ffn_out: Tensor,
    ones: Tensor,
}

fn alloc_scratch(backend: &Backend, hidden: u32, intermediate: u32) -> Scratch {
    use flint_tensor::DType;
    let z = |shape: &[u32]| backend.zero_tensor(shape);
    let pos_ids = Tensor::new(
        backend.storage(MAX_TOKENS as u64 * 4),
        vec![MAX_TOKENS],
        DType::U32,
    );
    backend.write_u32(&pos_ids.buf, &(0..MAX_TOKENS).collect::<Vec<_>>());
    let zero_ids = Tensor::new(
        backend.storage(MAX_TOKENS as u64 * 4),
        vec![MAX_TOKENS],
        DType::U32,
    );
    backend.write_u32(&zero_ids.buf, &vec![0u32; MAX_TOKENS as usize]);
    Scratch {
        ids: Tensor::new(
            backend.storage(MAX_TOKENS as u64 * 4),
            vec![MAX_TOKENS],
            DType::U32,
        ),
        pos_ids,
        zero_ids,
        meta: step::row_meta(backend),
        hidden: z(&[MAX_TOKENS, hidden]),
        normed: z(&[MAX_TOKENS, hidden]),
        q_out: z(&[MAX_TOKENS, hidden]),
        k_out: z(&[MAX_TOKENS, hidden]),
        v_out: z(&[MAX_TOKENS, hidden]),
        attn_out: z(&[MAX_TOKENS, hidden]),
        int_out: z(&[MAX_TOKENS, intermediate]),
        act_out: z(&[MAX_TOKENS, intermediate]),
        ffn_out: z(&[MAX_TOKENS, hidden]),
        ones: backend.tensor_f32(&vec![1.0; intermediate as usize], vec![intermediate]),
    }
}

pub struct Bert {
    cfg: Config,
    embed: Weight,
    pos_embed: Weight,
    type_embed: Option<Weight>,
    embed_ln: Tensor,
    embed_ln_bias: Tensor,
    layers: Vec<LayerW>,
    kv: KvPool,
    s: Scratch,
}

impl Bert {
    pub fn load(source: &dyn Checkpoint, backend: &Backend) -> Result<Self> {
        let cfg = Config::parse(&source.config_json()?)?;
        let mut w = loader::load_weights(backend, source, &PLAN)?;
        let embed = w.take("embeddings.word_embeddings.weight")?;
        let pos_embed = w.take("embeddings.position_embeddings.weight")?;
        let type_embed = w
            .has("embeddings.token_type_embeddings.weight")
            .then(|| w.take("embeddings.token_type_embeddings.weight"))
            .transpose()?;
        let embed_ln = w.take_tensor("embeddings.LayerNorm.weight")?;
        let embed_ln_bias = w.take_tensor("embeddings.LayerNorm.bias")?;
        let layers = (0..cfg.layers)
            .map(|i| take_layer(&mut w, i as usize))
            .collect::<Result<Vec<_>>>()?;
        let kv = KvPool::new(backend, cfg.heads, &[MAX_TOKENS], cfg.head_dim);
        let s = alloc_scratch(backend, cfg.hidden, cfg.intermediate);
        Ok(Self {
            cfg,
            embed,
            pos_embed,
            type_embed,
            embed_ln,
            embed_ln_bias,
            layers,
            kv,
            s,
        })
    }
}

impl TextEmbedder for Bert {
    fn embed(&mut self, backend: &mut Backend, tokens: &[u32]) -> Result<Vec<f32>> {
        let n = tokens.len() as u32;
        if n == 0 || n > MAX_TOKENS {
            return Err(Error::Model(format!(
                "token count {n} outside [1, {MAX_TOKENS}]"
            )));
        }
        let mut ids = vec![0u32; MAX_TOKENS as usize];
        ids[..tokens.len()].copy_from_slice(tokens);
        backend.write_u32(&self.s.ids.buf, &ids);
        let positions: Vec<u32> = (0..n).collect();
        let slots = vec![0u32; n as usize];
        step::write_row_meta(backend, &self.s.meta, &positions, &slots, n);

        let cfg = &self.cfg;
        let mut enc = backend.encoder()?;
        {
            let mut commands = Commands::begin(&mut enc);
            let s = &self.s;
            ops::embed(
                backend,
                &mut commands,
                &s.ids,
                &self.embed,
                Binding::Full(&s.hidden),
                &ops::EmbedSpec {
                    rows: n,
                    dim: cfg.hidden,
                    scale: 1.0,
                    split: 0,
                },
            )?;
            ops::embed(
                backend,
                &mut commands,
                &s.pos_ids,
                &self.pos_embed,
                Binding::Full(&s.normed),
                &ops::EmbedSpec {
                    rows: n,
                    dim: cfg.hidden,
                    scale: 1.0,
                    split: 0,
                },
            )?;
            ops::add(
                backend,
                &mut commands,
                Binding::Full(&s.hidden),
                Binding::Full(&s.normed),
                Binding::Full(&s.hidden),
                n * cfg.hidden,
            )?;
            if let Some(te) = &self.type_embed {
                ops::embed(
                    backend,
                    &mut commands,
                    &s.zero_ids,
                    te,
                    Binding::Full(&s.normed),
                    &ops::EmbedSpec {
                        rows: n,
                        dim: cfg.hidden,
                        scale: 1.0,
                        split: 0,
                    },
                )?;
                ops::add(
                    backend,
                    &mut commands,
                    Binding::Full(&s.hidden),
                    Binding::Full(&s.normed),
                    Binding::Full(&s.hidden),
                    n * cfg.hidden,
                )?;
            }
            ops::norm(
                backend,
                &mut commands,
                &NormSpec::new(NormMode::Layer, n, cfg.hidden, 1e-12),
                Binding::Full(&s.hidden),
                &self.embed_ln,
                Binding::Full(&self.embed_ln_bias),
                Binding::Full(&s.hidden),
            )?;

            for lw in &self.layers {
                ops::norm(
                    backend,
                    &mut commands,
                    &NormSpec::new(NormMode::Layer, n, cfg.hidden, 1e-12),
                    Binding::Full(&s.hidden),
                    &lw.attn_ln,
                    Binding::Full(&lw.attn_ln_bias),
                    Binding::Full(&s.normed),
                )?;
                ops::gemm_qkv(
                    backend,
                    &mut commands,
                    Binding::Full(&s.normed),
                    &ops::QkvSpec {
                        wq: &lw.q,
                        wk: &lw.k,
                        wv: &lw.v,
                        yq: Binding::Full(&s.q_out),
                        yk: Binding::Full(&s.k_out),
                        yv: Binding::Full(&s.v_out),
                        rows: n,
                        kv_width: cfg.hidden,
                    },
                )?;
                ops::bias(backend, &mut commands, Binding::Full(&s.q_out), &lw.q_bias, n, cfg.hidden)?;
                ops::bias(backend, &mut commands, Binding::Full(&s.k_out), &lw.k_bias, n, cfg.hidden)?;
                ops::bias(backend, &mut commands, Binding::Full(&s.v_out), &lw.v_bias, n, cfg.hidden)?;
                ops::kv_store(
                    backend,
                    &mut commands,
                    Binding::Full(&s.k_out),
                    Binding::Full(&s.v_out),
                    &self.kv,
                    n,
                    &s.meta,
                )?;
                ops::attn(
                    backend,
                    &mut commands,
                    Binding::Full(&s.q_out),
                    &self.kv,
                    Binding::Full(&s.attn_out),
                    &ops::AttnSpec {
                        q_heads: cfg.heads,
                        window: 0,
                        scale: (cfg.head_dim as f32).sqrt().recip(),
                        m: n,
                        causal: false,
                        slot: 0,
                        args: Binding::Full(&s.meta),
                    },
                )?;
                ops::gemm(
                    backend,
                    &mut commands,
                    Binding::Full(&s.attn_out),
                    &lw.o,
                    Binding::Full(&s.ffn_out),
                    n,
                )?;
                ops::bias(backend, &mut commands, Binding::Full(&s.ffn_out), &lw.o_bias, n, cfg.hidden)?;
                ops::add(
                    backend,
                    &mut commands,
                    Binding::Full(&s.hidden),
                    Binding::Full(&s.ffn_out),
                    Binding::Full(&s.hidden),
                    n * cfg.hidden,
                )?;
                ops::norm(
                    backend,
                    &mut commands,
                    &NormSpec::new(NormMode::Layer, n, cfg.hidden, 1e-12),
                    Binding::Full(&s.hidden),
                    &lw.ffn_ln,
                    Binding::Full(&lw.ffn_ln_bias),
                    Binding::Full(&s.normed),
                )?;
                ops::gemm(
                    backend,
                    &mut commands,
                    Binding::Full(&s.normed),
                    &lw.int,
                    Binding::Full(&s.int_out),
                    n,
                )?;
                ops::bias(
                    backend,
                    &mut commands,
                    Binding::Full(&s.int_out),
                    &lw.int_bias,
                    n,
                    cfg.intermediate,
                )?;
                ops::swiglu(
                    backend,
                    &mut commands,
                    Binding::Full(&s.int_out),
                    Binding::Full(&s.ones),
                    Binding::Full(&s.act_out),
                    n * cfg.intermediate,
                    Act::GeluTanh,
                )?;
                ops::gemm(
                    backend,
                    &mut commands,
                    Binding::Full(&s.act_out),
                    &lw.out,
                    Binding::Full(&s.ffn_out),
                    n,
                )?;
                ops::bias(backend, &mut commands, Binding::Full(&s.ffn_out), &lw.out_bias, n, cfg.hidden)?;
                ops::add(
                    backend,
                    &mut commands,
                    Binding::Full(&s.hidden),
                    Binding::Full(&s.ffn_out),
                    Binding::Full(&s.hidden),
                    n * cfg.hidden,
                )?;
            }
        }
        backend.submit(&mut enc)?;

        let rows = backend.read_f32(&self.s.hidden.buf, 0, (n * cfg.hidden) as usize)?;
        let mut out = vec![0.0f32; cfg.hidden as usize];
        for r in 0..n as usize {
            let row = &rows[r * cfg.hidden as usize..(r + 1) * cfg.hidden as usize];
            for (o, v) in out.iter_mut().zip(row) {
                *o += v / n as f32;
            }
        }
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        for v in &mut out {
            *v /= norm;
        }
        Ok(out)
    }
}

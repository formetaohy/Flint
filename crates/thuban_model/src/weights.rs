use thuban_backend::Backend;
use thuban_error::Result;
use thuban_tensor::{Tensor, Weight};

use crate::loader::WeightSet;

pub struct SwigluMlp {
    pub norm: Tensor,
    pub norm_bias: Option<Tensor>,
    pub gate: Weight,
    pub up: Weight,
    pub down: Weight,
}

pub fn pack_weights(backend: &Backend, ws: Vec<Weight>) -> Vec<Weight> {
    let tensors: Vec<&Tensor> = ws.iter().map(Weight::tensor).collect();
    let packed = backend.pack_weights(&tensors);
    let mut off = 0u64;
    ws.into_iter()
        .map(|w| {
            let t = w.tensor();
            let view = Tensor::view(packed.buf.clone(), off, t.shape.clone(), t.dtype);
            off += t.byte_len();
            match w {
                Weight::Plain(_) => Weight::plain(view),
                Weight::Quantized(_) => Weight::quantized(view),
            }
        })
        .collect()
}

pub fn take_mlp(
    w: &mut WeightSet,
    prefix: &str,
    layernorm: bool,
    backend: &Backend,
) -> Result<SwigluMlp> {
    let k = |n: &str| format!("{prefix}.{n}");
    let mut packed = pack_weights(
        backend,
        vec![
            w.take(&k("mlp.gate_proj.weight"))?,
            w.take(&k("mlp.up_proj.weight"))?,
        ],
    );
    Ok(SwigluMlp {
        norm: w.take_tensor(&k("post_attention_layernorm.weight"))?,
        norm_bias: if layernorm {
            Some(w.take_tensor(&k("post_attention_layernorm.bias"))?)
        } else {
            None
        },
        gate: packed.remove(0),
        up: packed.remove(0),
        down: w.take(&k("mlp.down_proj.weight"))?,
    })
}

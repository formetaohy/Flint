use flint_backend::{Backend, Binding, Commands};
use flint_error::Result;
use flint_kernel::{ATTN_BR, shader};
use flint_tensor::Tensor;

use crate::pool::KvPool;

pub struct AttnSpec<'a> {
    pub q_heads: u32,
    pub window: u32,
    pub scale: f32,
    pub m: u32,
    pub causal: bool,
    pub seq: u32,
    pub args: Binding<'a>,
}

pub fn attn(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    q: Binding<'_>,
    kv: &KvPool,
    y: Binding<'_>,
    spec: &AttnSpec<'_>,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::ATTN,
        &[
            ("M", spec.m as f64),
            ("N_HEADS", spec.q_heads as f64),
            ("HEAD_DIM", kv.head_dim as f64),
            ("POOL_LEN", kv.capacity as f64),
            ("SCALE", spec.scale as f64),
            ("WINDOW", spec.window as f64),
            ("NQ_PER_KV", (spec.q_heads / kv.kv_heads) as f64),
            ("SEQ", spec.seq as f64),
            ("CAUSAL", spec.causal as u32 as f64),
            ("MAX_PAGES", kv.max_pages as f64),
        ],
        &[
            q,
            Binding::Full(&kv.k),
            Binding::Full(&kv.v),
            y,
            spec.args,
            Binding::Full(&kv.block_table),
        ],
        [spec.m.div_ceil(ATTN_BR), spec.q_heads, 1],
    )
}

pub fn kv_store(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    k_src: Binding<'_>,
    v_src: Binding<'_>,
    kv: &KvPool,
    m: u32,
    meta: &Tensor,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::KV_STORE,
        &[
            ("N_KV", kv.kv_heads as f64),
            ("HEAD_DIM", kv.head_dim as f64),
            ("POOL_LEN", kv.capacity as f64),
            ("MAX_PAGES", kv.max_pages as f64),
        ],
        &[
            k_src,
            v_src,
            Binding::Full(&kv.k),
            Binding::Full(&kv.v),
            Binding::Full(meta),
            Binding::Full(&kv.block_table),
        ],
        [(kv.kv_heads * (kv.head_dim / 2)).div_ceil(256), m, 1],
    )
}

pub fn check_head_dim(head_dim: u32) -> Result<()> {
    if !(64..=512).contains(&head_dim) {
        return Err(flint_error::Error::Config(format!(
            "head_dim {head_dim} outside [64, 512]"
        )));
    }
    Ok(())
}

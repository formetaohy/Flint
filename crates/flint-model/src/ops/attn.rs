use flint_backend::{ATTN_BR, Backend, Binding, Commands, shader};
use flint_error::Result;
use flint_tensor::Tensor;

pub struct AttnSpec<'a> {
    pub q_heads: u32,
    pub window: u32,
    pub scale: f32,
    pub m: u32,
    pub args: &'a Tensor,
}

pub fn attn(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    q: Binding<'_>,
    kv: &crate::cache::KvCache,
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
            ("MAX_SEQ", kv.max_seq as f64),
            ("SCALE", spec.scale as f64),
            ("WINDOW", spec.window as f64),
            ("NQ_PER_KV", (spec.q_heads / kv.kv_heads) as f64),
        ],
        &[
            q,
            Binding::Full(&kv.k),
            Binding::Full(&kv.v),
            y,
            Binding::Full(spec.args),
        ],
        [spec.m.div_ceil(ATTN_BR), spec.q_heads, 1],
    )
}

pub fn kv_store(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    k_src: Binding<'_>,
    v_src: Binding<'_>,
    kv: &crate::cache::KvCache,
    m: u32,
    args: &Tensor,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::KV_STORE,
        &[
            ("N_KV", kv.kv_heads as f64),
            ("HEAD_DIM", kv.head_dim as f64),
            ("MAX_SEQ", kv.max_seq as f64),
        ],
        &[
            k_src,
            v_src,
            Binding::Full(&kv.k),
            Binding::Full(&kv.v),
            Binding::Full(args),
        ],
        [(kv.kv_heads * (kv.head_dim / 2)).div_ceil(256), m, 1],
    )
}

pub struct RepeatQkSpec {
    pub rows: u32,
    pub n_k: u32,
    pub n_v: u32,
    pub key_dim: u32,
    pub val_dim: u32,
}

pub fn repeat_qk(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    y: Binding<'_>,
    spec: &RepeatQkSpec,
) -> Result<()> {
    let conv_dim = 2 * spec.n_k * spec.key_dim + spec.n_v * spec.val_dim;
    backend.dispatch(
        commands,
        shader::REPEAT_QK,
        &[
            ("ROWS", spec.rows as f64),
            ("N_K", spec.n_k as f64),
            ("N_V", spec.n_v as f64),
            ("K_DIM", spec.key_dim as f64),
            ("RATIO", (spec.n_v / spec.n_k) as f64),
            ("CONV_DIM", conv_dim as f64),
        ],
        &[x, y],
        [spec.rows, 1, 1],
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

use thuban_backend::{Backend, Binding, Commands};
use thuban_error::Result;
use thuban_model::SeqChunk;
use thuban_model::ops::{self, Act, NormMode};
use thuban_model::pool::KvPool;
use thuban_model::weights::SwigluMlp;
use thuban_tensor::Tensor;

use super::config::Qwen35Config;
use super::scratch::Scratch;
use super::state::RecurrentPool;
use super::weights::{FullLayerW, LinearLayerW};

pub(super) struct FullCtx<'a> {
    pub(super) cfg: &'a Qwen35Config,
    pub(super) s: &'a Scratch,
    pub(super) cos: &'a Tensor,
    pub(super) sin: &'a Tensor,
}

pub(super) fn full_layer(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    ctx: &FullCtx<'_>,
    w: &FullLayerW,
    kv: &KvPool,
    m: u32,
    batch: &[SeqChunk],
) -> Result<()> {
    let (cfg, s) = (ctx.cfg, ctx.s);
    ops::norm(
        backend,
        commands,
        &ops::NormSpec::new(NormMode::Direct, m, cfg.hidden, cfg.norm_eps),
        Binding::Full(&s.hidden),
        &w.attn_norm,
        Binding::Full(&s.hidden),
        Binding::Full(&s.normed),
    )?;
    full_attn_block(backend, commands, ctx, w, kv, m, batch)?;
    ops::add(
        backend,
        commands,
        Binding::Full(&s.hidden),
        Binding::Full(&s.mlp.down_out),
        Binding::Full(&s.post_attn),
        m * cfg.hidden,
    )?;
    post_mlp(backend, commands, cfg, s, &w.mlp, m)
}

pub(super) fn linear_layer(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    ctx: &FullCtx<'_>,
    w: &LinearLayerW,
    state: &RecurrentPool,
    m: u32,
    batch: &[SeqChunk],
) -> Result<()> {
    let (cfg, s) = (ctx.cfg, ctx.s);
    ops::norm(
        backend,
        commands,
        &ops::NormSpec::new(NormMode::Direct, m, cfg.hidden, cfg.norm_eps),
        Binding::Full(&s.hidden),
        &w.attn_norm,
        Binding::Full(&s.hidden),
        Binding::Full(&s.normed),
    )?;
    linear_attn_block(backend, commands, ctx, w, state, m, batch)?;
    ops::add(
        backend,
        commands,
        Binding::Full(&s.hidden),
        Binding::Full(&s.mlp.down_out),
        Binding::Full(&s.post_attn),
        m * cfg.hidden,
    )?;
    post_mlp(backend, commands, cfg, s, &w.mlp, m)
}

fn full_attn_block(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    ctx: &FullCtx<'_>,
    w: &FullLayerW,
    kv: &KvPool,
    m: u32,
    batch: &[SeqChunk],
) -> Result<()> {
    let (cfg, s) = (ctx.cfg, ctx.s);
    let (nq, nkv, hd) = (cfg.q_heads, cfg.kv_heads, cfg.head_dim);

    if m == 1 {
        ops::gemv_qkv(
            backend,
            commands,
            Binding::Full(&s.normed),
            &ops::QkvSpec {
                wq: &w.q,
                wk: &w.k,
                wv: &w.v,
                yq: Binding::Full(&s.qg),
                yk: Binding::Full(&s.k_raw),
                yv: Binding::Full(&s.v_raw),
                rows: m,
                kv_width: nkv * hd,
            },
        )?;
    } else {
        ops::gemm(
            backend,
            commands,
            Binding::Full(&s.normed),
            &w.q,
            Binding::Full(&s.qg),
            m,
        )?;
        ops::gemm(
            backend,
            commands,
            Binding::Full(&s.normed),
            &w.k,
            Binding::Full(&s.k_raw),
            m,
        )?;
        ops::gemm(
            backend,
            commands,
            Binding::Full(&s.normed),
            &w.v,
            Binding::Full(&s.v_raw),
            m,
        )?;
    }

    ops::split_q_gate(
        backend,
        commands,
        Binding::Full(&s.qg),
        Binding::Full(&s.q),
        Binding::Full(&s.gate),
        &ops::SplitQGateSpec {
            rows: m,
            heads: nq,
            head_dim: hd,
        },
    )?;
    ops::norm(
        backend,
        commands,
        &ops::NormSpec::new(NormMode::Direct, m * nq, hd, cfg.norm_eps),
        Binding::Full(&s.q),
        &w.q_norm,
        Binding::Full(&s.q),
        Binding::Full(&s.q_normed),
    )?;
    ops::norm(
        backend,
        commands,
        &ops::NormSpec::new(NormMode::Direct, m * nkv, hd, cfg.norm_eps),
        Binding::Full(&s.k_raw),
        &w.k_norm,
        Binding::Full(&s.k_raw),
        Binding::Full(&s.k_normed),
    )?;

    let rope = ops::RopeInputs {
        cos: ctx.cos,
        sin: ctx.sin,
        args: &s.meta,
    };
    ops::rope(
        backend,
        commands,
        Binding::Full(&s.q_normed),
        &rope,
        &ops::RopeArgs {
            heads: nq,
            head_dim: hd,
            rot: cfg.rotary_dim,
            m,
        },
    )?;
    ops::rope(
        backend,
        commands,
        Binding::Full(&s.k_normed),
        &rope,
        &ops::RopeArgs {
            heads: nkv,
            head_dim: hd,
            rot: cfg.rotary_dim,
            m,
        },
    )?;
    ops::kv_store(
        backend,
        commands,
        Binding::Full(&s.k_normed),
        Binding::Full(&s.v_raw),
        kv,
        m,
        &s.meta,
    )?;

    let qw = nq * hd;
    let mut row_off = 0u32;
    for chunk in batch {
        let m_s = chunk.len();
        let span = m_s as u64 * qw as u64 * 4;
        ops::attn(
            backend,
            commands,
            Binding::Slice(&s.q_normed, row_off as u64 * qw as u64 * 4, span),
            kv,
            Binding::Slice(&s.attn_out, row_off as u64 * qw as u64 * 4, span),
            &ops::AttnSpec {
                q_heads: nq,
                window: 0,
                scale: cfg.attn_scale,
                m: m_s,
                causal: true,
                seq: chunk.seq,
                args: Binding::Slice(&s.meta, row_off as u64 * 32, m_s as u64 * 32),
            },
        )?;
        row_off += m_s;
    }
    ops::sigmoid_mul(
        backend,
        commands,
        Binding::Full(&s.attn_out),
        Binding::Full(&s.gate),
        Binding::Full(&s.attn_gated),
        m * nq * hd,
    )?;
    ops::gemm(
        backend,
        commands,
        Binding::Full(&s.attn_gated),
        &w.o,
        Binding::Full(&s.mlp.down_out),
        m,
    )
}

fn linear_attn_block(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    ctx: &FullCtx<'_>,
    w: &LinearLayerW,
    state: &RecurrentPool,
    m: u32,
    batch: &[SeqChunk],
) -> Result<()> {
    let (cfg, s) = (ctx.cfg, ctx.s);
    let heads = cfg.lin_val_heads;
    let (key_d, val_d, conv_d) = (cfg.key_dim(), cfg.value_dim(), cfg.conv_dim());

    ops::gemm(
        backend,
        commands,
        Binding::Full(&s.normed),
        &w.in_proj_qkv,
        Binding::Full(&s.qkv_proj),
        m,
    )?;
    ops::gemm(
        backend,
        commands,
        Binding::Full(&s.normed),
        &w.in_proj_z,
        Binding::Full(&s.z),
        m,
    )?;
    ops::gemm(
        backend,
        commands,
        Binding::Full(&s.normed),
        &w.in_proj_b,
        Binding::Full(&s.b),
        m,
    )?;
    ops::gemm(
        backend,
        commands,
        Binding::Full(&s.normed),
        &w.in_proj_a,
        Binding::Full(&s.a),
        m,
    )?;

    let row = |t: u32, stride: u32| t as u64 * stride as u64 * 4;
    let mut t_off = 0u32;
    for chunk in batch {
        let seq = chunk.seq;
        for _ in 0..chunk.len() {
            let t = t_off;
            ops::conv1d(
                backend,
                commands,
                Binding::Slice(&s.qkv_proj, row(t, conv_d), conv_d as u64 * 4),
                &w.conv1d,
                state.conv_slice(seq),
                Binding::Slice(&s.conv_out, row(t, conv_d), conv_d as u64 * 4),
                &ops::ConvSpec { dim: conv_d },
            )?;
            t_off += 1;
        }
    }
    ops::repeat_qk(
        backend,
        commands,
        Binding::Full(&s.conv_out),
        Binding::Full(&s.qk_expanded),
        &ops::RepeatQkSpec {
            rows: m,
            key_heads: cfg.lin_key_heads,
            val_heads: heads,
            key_dim: cfg.lin_key_dim,
            conv_dim: conv_d,
        },
    )?;
    let exp_d = cfg.qk_exp_dim();

    let qkb = exp_d as u64 * 2;
    let mut t_off = 0u32;
    for chunk in batch {
        let seq = chunk.seq;
        for _ in 0..chunk.len() {
            let t = t_off;
            ops::delta_gate(
                backend,
                commands,
                &ops::DeltaGate {
                    b: Binding::Full(&s.b),
                    a: Binding::Full(&s.a),
                    a_log: &w.a_log,
                    dt_bias: &w.dt_bias,
                    beta: Binding::Full(&s.beta),
                    g: Binding::Full(&s.g),
                    heads,
                    row: t,
                },
            )?;
            let ed = row(t, exp_d);
            let (kb, vb) = (key_d as u64 * 4, val_d as u64 * 4);
            ops::delta_recur(
                backend,
                commands,
                &ops::DeltaRecur {
                    q: Binding::Slice(&s.qk_expanded, ed, qkb),
                    k: Binding::Slice(&s.qk_expanded, ed + qkb, qkb),
                    v: Binding::Slice(&s.conv_out, row(t, conv_d) + kb * 2, vb),
                    beta: Binding::Full(&s.beta),
                    g: Binding::Full(&s.g),
                    state: state.recur_slice(seq),
                    y: Binding::Slice(&s.attn_out, row(t, val_d), vb),
                    heads,
                    key_dim: cfg.lin_key_dim,
                    val_dim: cfg.lin_val_dim,
                },
            )?;
            t_off += 1;
        }
    }

    let span = m as u64 * val_d as u64 * 4;
    ops::norm(
        backend,
        commands,
        &ops::NormSpec::new(NormMode::Gated, m * heads, cfg.lin_val_dim, cfg.norm_eps),
        Binding::Slice(&s.attn_out, 0, span),
        &w.norm,
        Binding::Slice(&s.z, 0, span),
        Binding::Slice(&s.attn_gated, 0, span),
    )?;
    ops::gemm(
        backend,
        commands,
        Binding::Full(&s.attn_gated),
        &w.out_proj,
        Binding::Full(&s.mlp.down_out),
        m,
    )
}

fn post_mlp(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    cfg: &Qwen35Config,
    s: &Scratch,
    mlp: &SwigluMlp,
    m: u32,
) -> Result<()> {
    ops::norm(
        backend,
        commands,
        &ops::NormSpec::new(NormMode::Direct, m, cfg.hidden, cfg.norm_eps),
        Binding::Full(&s.post_attn),
        &mlp.norm,
        Binding::Full(&s.post_attn),
        Binding::Full(&s.normed),
    )?;
    ops::swiglu_mlp(
        backend,
        commands,
        Binding::Full(&s.normed),
        mlp,
        &s.mlp,
        Binding::Full(&s.mlp.down_out),
        &ops::MlpSpec {
            rows: m,
            intermediate: cfg.intermediate,
            act: Act::Silu,
            acc: false,
        },
    )?;
    ops::add(
        backend,
        commands,
        Binding::Full(&s.post_attn),
        Binding::Full(&s.mlp.down_out),
        Binding::Full(&s.hidden),
        m * cfg.hidden,
    )
}

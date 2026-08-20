use flint_backend::{Backend, Binding, Commands};
use flint_error::Result;
use flint_kernel::shader;
use flint_tensor::{DType, Tensor};

use crate::ops::{gemm, swiglu};
use crate::routing::Routing;
use crate::traits::MAX_M;
use crate::weights::{ExpertWeights, MoeMlp};
use flint_kernel::Act;

pub struct MoeTiles {
    pub logits: Tensor,

    pub packed: Vec<Tensor>,
    pub gate: Tensor,
    pub up: Tensor,
    pub act: Tensor,
    pub down: Tensor,

    pub acc: Tensor,

    pub rows: Tensor,
    pub weights: Tensor,
}

impl MoeTiles {
    pub fn new(cfg: &MoeTilesConfig, backend: &Backend) -> Self {
        let z = |shape: &[u32]| backend.zero_tensor(shape, DType::F32);

        let pairs = (cfg.experts + 2) * 64 + cfg.rows * cfg.top_k + cfg.rows;
        MoeTiles {
            logits: z(&[MAX_M, cfg.experts]),
            packed: (0..=cfg.experts).map(|_| z(&[MAX_M, cfg.hidden])).collect(),
            gate: z(&[MAX_M, cfg.intermediate]),
            up: z(&[MAX_M, cfg.intermediate]),
            act: z(&[MAX_M, cfg.intermediate]),
            down: z(&[MAX_M, cfg.hidden]),
            acc: z(&[MAX_M, cfg.hidden]),
            rows: Tensor::new(backend.storage(pairs as u64 * 4), vec![pairs], DType::U32),
            weights: z(&[pairs]),
        }
    }
}

pub struct MoeTilesConfig {
    pub experts: u32,
    pub rows: u32,
    pub top_k: u32,
    pub hidden: u32,
    pub intermediate: u32,
}

pub struct MoeSpec {
    pub intermediate: u32,
    pub act: Act,
    pub hidden: u32,
}

pub fn moe_apply(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    moe: &MoeMlp,
    t: &MoeTiles,
    r: &Routing,
    spec: &MoeSpec,
) -> Result<()> {
    for e in 0..=moe.experts.len() {
        let c = r.count(e);
        if c == 0 {
            continue;
        }

        let packed = if e < moe.experts.len() {
            let off = r.offset(e);
            expert_gather(
                backend,
                commands,
                x,
                Binding::Slice(&t.rows, off, c as u64 * 4),
                Binding::Full(&t.packed[e]),
                &GatherSpec {
                    hidden: spec.hidden,
                    count: c,
                },
            )?;
            Binding::Full(&t.packed[e])
        } else {
            x
        };
        let ew = if e < moe.experts.len() {
            &moe.experts[e]
        } else {
            moe.shared.as_ref().expect("shared expert present")
        };
        expert_mlp(backend, commands, packed, ew, t, spec, c)?;
        let off = r.offset(e);
        expert_scatter(
            backend,
            commands,
            Binding::Full(&t.acc),
            Binding::Full(&t.down),
            Binding::Slice(&t.rows, off, c as u64 * 4),
            Binding::Slice(&t.weights, off, c as u64 * 4),
            &GatherSpec {
                hidden: spec.hidden,
                count: c,
            },
        )?;
    }
    Ok(())
}

fn expert_mlp(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    ew: &ExpertWeights,
    t: &MoeTiles,
    spec: &MoeSpec,
    count: u32,
) -> Result<()> {
    let y_gate = if count == 1 {
        Binding::Slice(&t.gate, 0, spec.intermediate as u64 * 4)
    } else {
        Binding::Full(&t.gate)
    };
    let y_up = if count == 1 {
        Binding::Slice(&t.up, 0, spec.intermediate as u64 * 4)
    } else {
        Binding::Full(&t.up)
    };
    gemm(backend, commands, x, &ew.gate, y_gate, count)?;
    gemm(backend, commands, x, &ew.up, y_up, count)?;
    swiglu(
        backend,
        commands,
        Binding::Full(&t.gate),
        Binding::Full(&t.up),
        Binding::Full(&t.act),
        count * spec.intermediate,
        spec.act,
    )?;
    gemm(
        backend,
        commands,
        Binding::Full(&t.act),
        &ew.down,
        Binding::Full(&t.down),
        count,
    )
}

pub struct GatherSpec {
    pub hidden: u32,
    pub count: u32,
}

pub fn expert_gather(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    ids: Binding<'_>,
    out: Binding<'_>,
    spec: &GatherSpec,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::EXPERT_GATHER,
        &[("HIDDEN", spec.hidden as f64), ("COUNT", spec.count as f64)],
        &[x, ids, out],
        [(spec.count * spec.hidden).div_ceil(256), 1, 1],
    )
}

pub fn expert_scatter(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    acc: Binding<'_>,
    src: Binding<'_>,
    ids: Binding<'_>,
    weights: Binding<'_>,
    spec: &GatherSpec,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::EXPERT_SCATTER,
        &[("HIDDEN", spec.hidden as f64), ("COUNT", spec.count as f64)],
        &[acc, src, ids, weights],
        [(spec.count * spec.hidden).div_ceil(256), 1, 1],
    )
}

pub fn zero_rows(
    backend: &mut Backend,
    commands: &mut Commands<'_>,
    x: Binding<'_>,
    n: u32,
) -> Result<()> {
    backend.dispatch(
        commands,
        shader::ZERO_ROWS,
        &[("N_ELEM", n as f64)],
        &[x],
        [n.div_ceil(256), 1, 1],
    )
}

use flint_backend::{Backend, Binding, Pass};
use flint_error::Result;
use flint_kernel::name;
use flint_tensor::{DType, Tensor};

use crate::blocks::{ExpertWeights, MoeMlp};
use crate::ops::{Act, M_MAX, gemm, swiglu};
use crate::routing::Routing;

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
        let z = |shape: &[u32]| backend.zero_tensor(shape);

        let pairs = (cfg.experts + 2) * 64 + cfg.rows * cfg.top_k + cfg.rows;
        MoeTiles {
            logits: z(&[M_MAX, cfg.experts]),
            packed: (0..=cfg.experts).map(|_| z(&[M_MAX, cfg.hidden])).collect(),
            gate: z(&[M_MAX, cfg.intermediate]),
            up: z(&[M_MAX, cfg.intermediate]),
            act: z(&[M_MAX, cfg.intermediate]),
            down: z(&[M_MAX, cfg.hidden]),
            acc: z(&[M_MAX, cfg.hidden]),
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
pub fn moe_apply(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    moe: &MoeMlp,
    t: &MoeTiles,
    r: &Routing,
    intermediate: u32,
    act: Act,
    hidden: u32,
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
                pass,
                x,
                Binding::Slice(&t.rows, off, c as u64 * 4),
                Binding::Full(&t.packed[e]),
                hidden,
                c,
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
        expert_mlp(backend, pass, packed, ew, t, intermediate, act, c)?;
        let off = r.offset(e);
        expert_scatter(
            backend,
            pass,
            Binding::Full(&t.acc),
            Binding::Full(&t.down),
            Binding::Slice(&t.rows, off, c as u64 * 4),
            Binding::Slice(&t.weights, off, c as u64 * 4),
            hidden,
            c,
        )?;
    }
    Ok(())
}
fn expert_mlp(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    ew: &ExpertWeights,
    t: &MoeTiles,
    intermediate: u32,
    act: Act,
    count: u32,
) -> Result<()> {
    let rows = count;
    let y_gate = if count == 1 {
        Binding::Slice(&t.gate, 0, intermediate as u64 * 4)
    } else {
        Binding::Full(&t.gate)
    };
    let y_up = if count == 1 {
        Binding::Slice(&t.up, 0, intermediate as u64 * 4)
    } else {
        Binding::Full(&t.up)
    };
    gemm(backend, pass, x, &ew.gate, y_gate, rows)?;
    gemm(backend, pass, x, &ew.up, y_up, rows)?;
    swiglu(
        backend,
        pass,
        Binding::Full(&t.gate),
        Binding::Full(&t.up),
        Binding::Full(&t.act),
        count * intermediate,
        act,
    )?;
    gemm(
        backend,
        pass,
        Binding::Full(&t.act),
        &ew.down,
        Binding::Full(&t.down),
        rows,
    )
}

pub fn expert_gather(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    x: Binding<'_>,
    ids: Binding<'_>,
    out: Binding<'_>,
    hidden: u32,
    count: u32,
) -> Result<()> {
    backend.dispatch(
        pass,
        name::EXPERT_GATHER,
        &[("HIDDEN", hidden as f64), ("COUNT", count as f64)],
        &[x, ids, out],
        [(count * hidden).div_ceil(256), 1, 1],
    )
}
pub fn expert_scatter(
    backend: &mut Backend,
    pass: &mut Pass<'_>,
    acc: Binding<'_>,
    src: Binding<'_>,
    ids: Binding<'_>,
    weights: Binding<'_>,
    hidden: u32,
    count: u32,
) -> Result<()> {
    backend.dispatch(
        pass,
        name::EXPERT_SCATTER,
        &[("HIDDEN", hidden as f64), ("COUNT", count as f64)],
        &[acc, src, ids, weights],
        [(count * hidden).div_ceil(256), 1, 1],
    )
}

pub fn zero_rows(backend: &mut Backend, pass: &mut Pass<'_>, x: Binding<'_>, n: u32) -> Result<()> {
    backend.dispatch(
        pass,
        name::ZERO_ROWS,
        &[("N_ELEM", n as f64)],
        &[x],
        [n.div_ceil(256), 1, 1],
    )
}

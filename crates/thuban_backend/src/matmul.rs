use thuban_error::Result;
use thuban_gpu::CoopVariant;
use thuban_tensor::Weight;

use crate::{Backend, Binding, Commands};
use thuban_kernel::shader;

impl Backend {
    fn ensure_gemm_partial(&mut self, words: u32) {
        if words > self.gemm_partial.numel() as u32 {
            let old = std::mem::replace(
                &mut self.gemm_partial,
                Self::partial_buf(self.device.as_ref(), words as usize)
                    .expect("gemm partial growth"),
            );
            self.retire(old);
        }
    }

    fn ensure_gemm_x_f16(&mut self, words: u32) {
        if words > self.gemm_x_f16.numel() as u32 {
            let old = std::mem::replace(
                &mut self.gemm_x_f16,
                Self::partial_f16_buf(self.device.as_ref(), words as usize)
                    .expect("gemm f16 staging growth"),
            );
            self.retire(old);
        }
    }

    fn weight_io(w: &Weight) -> (u32, u32, Binding<'_>, u32) {
        assert_eq!(
            w.tensor().shape.len(),
            2,
            "gemm weight must be a [N, K] matrix"
        );
        let (n, k) = (w.tensor().shape[0], w.tensor().shape[1]);
        let qtype = match w.tensor().dtype {
            thuban_tensor::DType::F32 => 0,
            thuban_tensor::DType::F16 => 1,
            thuban_tensor::DType::Bf16 => 30,
            thuban_tensor::DType::Quant(q) => q.as_u32(),
            thuban_tensor::DType::U32 => {
                unreachable!("gemm operands are weights, never index tensors")
            }
        };
        (n, k, Binding::Full(w.tensor()), qtype)
    }

    pub fn gemm_acc(
        &mut self,
        commands: &mut Commands<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
        rows: u32,
        acc: bool,
    ) -> Result<()> {
        let (n, k, wb, qtype) = Self::weight_io(w);
        assert!(
            k.is_multiple_of(32),
            "gemm K {k} is not a multiple of the BK=32 tile"
        );
        let main = rows - rows % 128;
        let coop = if main > 0 && n.is_multiple_of(128) {
            self.device.coop_gemm().map(|v| match v {
                CoopVariant::M16 => shader::GEMM_COOP_M16,
                CoopVariant::M8 => shader::GEMM_COOP_M8,
            })
        } else {
            None
        };
        if let Some(kernel) = coop {
            self.ensure_gemm_x_f16(main * k);
            let x_f16 = Binding::Slice(&self.gemm_x_f16, 0, main as u64 * k as u64 * 2);
            self.dispatch(
                commands,
                shader::TO_F16,
                &[("N_ELEM", (main * k) as f64)],
                &[x, x_f16],
                [(main * k / 4).div_ceil(256), 1, 1],
            )?;
            let consts = [
                ("N", n as f64),
                ("K", k as f64),
                ("M", main as f64),
                ("SEGS", 1.0),
                ("QTYPE", qtype as f64),
                ("ACC", acc as u32 as f64),
                ("Y_STRIDE", n as f64),
                ("Y_OFF", 0.0),
            ];
            let lut = Binding::Full(self.quant_lut());
            self.dispatch(
                commands,
                kernel,
                &consts,
                &[x_f16, wb, lut, y],
                [n.div_ceil(128), main.div_ceil(128), 1],
            )?;
            if rows > main {
                let tail = rows - main;
                let xt = x.sub_slice(main as u64 * k as u64 * 4, tail as u64 * k as u64 * 4);
                let yt = y.sub_slice(main as u64 * n as u64 * 4, tail as u64 * n as u64 * 4);
                self.gemm_tiled(commands, xt, w, yt, tail, acc)?;
            }
            return Ok(());
        }
        self.gemm_tiled(commands, x, w, y, rows, acc)
    }

    fn gemm_tiled(
        &mut self,
        commands: &mut Commands<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
        rows: u32,
        acc: bool,
    ) -> Result<()> {
        let (n, k, wb, qtype) = Self::weight_io(w);
        let segs = if rows > 1 && k >= 8192 && k.is_multiple_of(128) {
            4
        } else {
            1
        };
        let gemm_acc = if segs > 1 { 0 } else { acc as u32 };
        let consts = [
            ("N", n as f64),
            ("K", k as f64),
            ("M", rows as f64),
            ("QTYPE", qtype as f64),
            ("ACC", gemm_acc as f64),
            ("Y_STRIDE", n as f64),
            ("Y_OFF", 0.0),
            ("SEGS", segs as f64),
        ];
        let yb = if segs > 1 {
            self.ensure_gemm_partial(segs * rows * n);
            Binding::Slice(&self.gemm_partial, 0, (segs * rows * n) as u64 * 4)
        } else {
            y
        };
        let lut = Binding::Full(self.quant_lut());
        let bufs = [x, wb, lut, yb];
        self.dispatch(
            commands,
            shader::GEMM,
            &consts,
            &bufs,
            [n.div_ceil(128), rows.div_ceil(64), segs],
        )?;
        if segs > 1 {
            let mconsts = [
                ("M", rows as f64),
                ("N", n as f64),
                ("Y_STRIDE", n as f64),
                ("Y_OFF", 0.0),
                ("SEGS", segs as f64),
                ("ACC", acc as u32 as f64),
            ];
            let bufs = [
                Binding::Slice(&self.gemm_partial, 0, (segs * rows * n) as u64 * 4),
                y,
            ];
            self.dispatch(
                commands,
                shader::MERGE_GEMM,
                &mconsts,
                &bufs,
                [n.div_ceil(256), 1, 1],
            )?;
        }
        Ok(())
    }
}

pub struct GemvOp<'a> {
    pub w: &'a Weight,
    pub y: Binding<'a>,
    pub acc: bool,
}

impl Backend {
    pub fn gemv(
        &mut self,
        commands: &mut Commands<'_>,
        x: Binding<'_>,
        ops: &[GemvOp<'_>],
    ) -> Result<()> {
        assert!(
            !ops.is_empty() && ops.len() <= 3,
            "gemv takes between 1 and 3 ops"
        );
        let mut ns = [0u32; 3];
        let mut ks = [0u32; 3];
        let mut qts = [0u32; 3];
        let mut acs = [0u32; 3];
        let mut offs = [0u32; 3];
        let mut max_base = 0u32;
        let mut wb: Option<Binding<'_>> = None;
        let mut base_offset = 0u64;
        for (i, op) in ops.iter().enumerate() {
            let t = op.w.tensor();
            let (n, k, binding, qtype) = Self::weight_io(op.w);
            assert!(k.is_multiple_of(32), "gemv K {k} is not a multiple of 32");
            if let Some(prev) = &wb {
                let Binding::Full(pt) = prev else {
                    unreachable!("gemv weights bind in full")
                };
                assert!(
                    pt.buf.same(&t.buf),
                    "gemv ops must share one packed weight buffer"
                );
            } else {
                wb = Some(binding);
                base_offset = t.offset;
            }
            ns[i] = n;
            ks[i] = k;
            qts[i] = qtype;
            acs[i] = op.acc as u32;
            offs[i] = (t.offset - base_offset) as u32;
            max_base = max_base.max(n.div_ceil(8));
        }
        let consts = [
            ("N0", ns[0] as f64),
            ("N1", ns[1] as f64),
            ("N2", ns[2] as f64),
            ("K0", ks[0] as f64),
            ("K1", ks[1] as f64),
            ("K2", ks[2] as f64),
            ("QT0", qts[0] as f64),
            ("QT1", qts[1] as f64),
            ("QT2", qts[2] as f64),
            ("AC0", acs[0] as f64),
            ("AC1", acs[1] as f64),
            ("AC2", acs[2] as f64),
            ("O0", offs[0] as f64),
            ("O1", offs[1] as f64),
            ("O2", offs[2] as f64),
        ];
        let lut = Binding::Full(self.quant_lut());
        let dummy = Binding::Full(self.dummy());
        let (mut y0, mut y1, mut y2) = (dummy, dummy, dummy);
        for (i, op) in ops.iter().enumerate() {
            match i {
                0 => y0 = op.y,
                1 => y1 = op.y,
                _ => y2 = op.y,
            }
        }
        let bufs = [x, wb.expect("ops implies a weight"), y0, y1, y2, lut];
        self.dispatch(commands, shader::GEMV, &consts, &bufs, [max_base, 1, 3])
    }
}

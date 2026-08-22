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

    fn ensure_gemm_xf16(&mut self, words: u32) {
        if words > self.gemm_xf16.numel() as u32 {
            let old = std::mem::replace(
                &mut self.gemm_xf16,
                Self::partial_f16_buf(self.device.as_ref(), words as usize)
                    .expect("gemm f16 staging growth"),
            );
            self.retire(old);
        }
    }

    fn ensure_gemv_partial(&mut self, words: u32) {
        if words > self.gemv_partial.numel() as u32 {
            let old = std::mem::replace(
                &mut self.gemv_partial,
                Self::partial_buf(self.device.as_ref(), words as usize)
                    .expect("gemv partial growth"),
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
                CoopVariant::M16 => shader::GEMM_COOP,
                CoopVariant::M8 => shader::GEMM_COOP8,
            })
        } else {
            None
        };
        if let Some(kernel) = coop {
            self.ensure_gemm_xf16(main * k);
            let xf = Binding::Slice(&self.gemm_xf16, 0, main as u64 * k as u64 * 2);
            Self::set(
                &self.kernels,
                commands,
                shader::TO_F16,
                &[("N_ELEM", (main * k) as f64)],
                &[x, xf],
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
            Self::set(
                &self.kernels,
                commands,
                kernel,
                &consts,
                &[xf, wb, lut, y],
                [n.div_ceil(128), main.div_ceil(128), 1],
            )?;
            if rows > main {
                let tail = rows - main;
                let xt = x.sub_slice(main as u64 * k as u64 * 4, tail as u64 * k as u64 * 4);
                let yt = y.sub_slice(main as u64 * n as u64 * 4, tail as u64 * n as u64 * 4);
                self.classic_gemm(commands, xt, w, yt, tail, acc)?;
            }
            return Ok(());
        }
        self.classic_gemm(commands, x, w, y, rows, acc)
    }

    fn classic_gemm(
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
        Self::set(
            &self.kernels,
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
            Self::set(
                &self.kernels,
                commands,
                shader::MERGE_GEMM,
                &mconsts,
                &bufs,
                [n.div_ceil(256), 1, 1],
            )?;
        }
        Ok(())
    }

    pub fn gemv(
        &mut self,
        commands: &mut Commands<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
    ) -> Result<()> {
        self.gemv_acc(commands, x, w, y, false)
    }

    pub fn gemv_acc(
        &mut self,
        commands: &mut Commands<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
        acc: bool,
    ) -> Result<()> {
        let (n, k, wb, qtype) = Self::weight_io(w);
        assert!(k.is_multiple_of(32), "gemv K {k} is not a multiple of 32");
        let base = n.div_ceil(64);
        let segs: u32 = if base >= 96 {
            1
        } else {
            [8u32, 4, 2, 1]
                .into_iter()
                .find(|s| k % (*s * 32) == 0 && base * *s >= 96)
                .unwrap_or(1)
        };
        if segs > 1 {
            self.ensure_gemv_partial(n * segs);
        }
        let consts = [
            ("N", n as f64),
            ("K", k as f64),
            ("QTYPE", qtype as f64),
            ("SEGS", segs as f64),
            ("ACC", acc as u32 as f64),
        ];
        let out = if segs == 1 {
            y
        } else {
            Binding::Slice(&self.gemv_partial, 0, n as u64 * 4 * segs as u64)
        };
        let lut = Binding::Full(self.quant_lut());
        let bufs = [x, wb, lut, out];
        Self::set(
            &self.kernels,
            commands,
            shader::GEMV,
            &consts,
            &bufs,
            [base, segs, 1],
        )?;
        if segs > 1 {
            let bufs = [
                Binding::Slice(&self.gemv_partial, 0, n as u64 * 4 * segs as u64),
                y,
            ];
            Self::set(
                &self.kernels,
                commands,
                shader::MERGE_GEMV,
                &[
                    ("N", n as f64),
                    ("SEGS", segs as f64),
                    ("ACC", acc as u32 as f64),
                ],
                &bufs,
                [n.div_ceil(256), 1, 1],
            )?;
        }
        Ok(())
    }
}

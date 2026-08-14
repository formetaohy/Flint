use flint_error::Result;
use flint_tensor::{DType, Weight};

use crate::{Backend, Binding, Commands, shader};

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

    fn weight_io(w: &Weight) -> (u32, u32, Binding<'_>, DType) {
        assert_eq!(
            w.tensor().shape.len(),
            2,
            "gemm weight must be a [N, K] matrix"
        );
        let (n, k) = (w.tensor().shape[0], w.tensor().shape[1]);
        let dtype = match w.tensor().dtype {
            DType::Bf16 | DType::I8 => w.tensor().dtype,
            DType::F32 | DType::U32 => {
                unreachable!("gemm operands are weights, never index tensors")
            }
        };
        (n, k, Binding::Full(w.tensor()), dtype)
    }

    fn scale_binding<'a>(unit_scale: &'a flint_tensor::Tensor, w: &'a Weight) -> Binding<'a> {
        match w.scale() {
            Some(s) => Binding::Full(s),
            None => Binding::Full(unit_scale),
        }
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
        let (n, k, wb, dtype) = Self::weight_io(w);
        assert!(
            k.is_multiple_of(32),
            "gemm K {k} is not a multiple of the BK=32 tile"
        );
        let segs = if rows > 1 && k >= 8192 && k.is_multiple_of(128) {
            4
        } else {
            1
        };
        let gemm_acc = if segs > 1 { 0 } else { acc as u32 };
        let coop = dtype == DType::Bf16 && n.is_multiple_of(16) && rows.is_multiple_of(16);
        let kernel = if coop {
            shader::GEMM_COOP
        } else {
            shader::GEMM
        };
        let consts = [
            ("N", n as f64),
            ("K", k as f64),
            ("M", rows as f64),
            ("WDTYPE", dtype_flag(dtype)),
            ("GROUP", group_const(w)),
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
        let unit_scale = &self.unit_scale;
        let bufs = [x, wb, Self::scale_binding(unit_scale, w), yb];
        let tm = if coop { 64 } else { 32 };
        Self::set(
            &mut self.kernels,
            commands,
            kernel,
            &consts,
            &bufs,
            [n.div_ceil(32), rows.div_ceil(tm), segs],
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
                &mut self.kernels,
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
        let (n, k, wb, dtype) = Self::weight_io(w);
        let v4 = k.is_multiple_of(32);
        let kb = if v4 { k / 32 } else { k / 16 };
        let base = n.div_ceil(if v4 { 512 } else { 256 });
        let max_segs = if v4 {
            if n >= 4096 {
                16
            } else {
                64
            }
        } else {
            8
        };
        let wgs_target = if v4 { 128 } else { 96 };
        let segs: u32 = [64u32, 32, 16, 8, 4, 2, 1]
            .into_iter()
            .find(|s| *s <= max_segs && kb % *s == 0 && base * *s >= wgs_target)
            .unwrap_or(1);
        if segs > 1 {
            self.ensure_gemv_partial(n * segs);
        }
        let unit_scale = &self.unit_scale;
        let scale = Self::scale_binding(unit_scale, w);
        let consts = [
            ("N", n as f64),
            ("K", k as f64),
            ("WDTYPE", dtype_flag(dtype)),
            ("GROUP", group_const(w)),
            ("SEGS", segs as f64),
            ("ACC", acc as u32 as f64),
        ];
        let out = if segs == 1 {
            y
        } else {
            Binding::Slice(&self.gemv_partial, 0, n as u64 * 4 * segs as u64)
        };
        let bufs = [x, wb, scale, out];
        Self::set(
            &mut self.kernels,
            commands,
            if v4 { shader::GEMV_V4 } else { shader::GEMV },
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
                &mut self.kernels,
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

fn dtype_flag(dtype: DType) -> f64 {
    match dtype {
        DType::Bf16 => 0.0,
        DType::I8 => 1.0,
        DType::F32 | DType::U32 => unreachable!("gemm weights are bf16-packed or i8"),
    }
}

fn group_const(w: &Weight) -> f64 {
    match w.group() {
        Some(group) => group as f64,
        None => 0.0,
    }
}

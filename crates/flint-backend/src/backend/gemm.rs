use flint_error::Result;
use flint_kernel::name;
use flint_tensor::{DType, Tensor, Weight};

use super::{Backend, Binding, Pass};

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

    pub fn gemm(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
        rows: u32,
    ) -> Result<()> {
        let n = w.tensor().shape[0];
        self.gemm_strided(pass, x, w, y, rows, false, 0, n)
    }

    pub fn gemm_acc(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
        rows: u32,
        acc: bool,
    ) -> Result<()> {
        let n = w.tensor().shape[0];
        self.gemm_strided(pass, x, w, y, rows, acc, 0, n)
    }

    pub fn gemm_strided(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
        rows: u32,
        acc: bool,
        y_off: u32,
        y_stride: u32,
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
        let consts = [
            ("N", n as f64),
            ("K", k as f64),
            ("M", rows as f64),
            ("WDTYPE", dtype_flag(dtype)),
            ("GROUP", group_const(w)),
            ("ACC", gemm_acc as f64),
            ("Y_STRIDE", y_stride as f64),
            ("Y_OFF", y_off as f64),
        ];
        let consts = consts
            .into_iter()
            .chain([("SEGS", segs as f64)])
            .collect::<Vec<_>>();
        let (yb, yslice) = if segs > 1 {
            self.ensure_gemm_partial(segs * rows * y_stride);
            (
                Binding::Slice(&self.gemm_partial, 0, (segs * rows * y_stride) as u64 * 4),
                y,
            )
        } else {
            (y, y)
        };
        let bufs = [x, wb, Self::scale_binding(&self.dummy_scale, w), yb];
        Self::set(
            &self.kernels,
            pass,
            name::GEMM,
            &consts,
            &bufs,
            [n.div_ceil(32), rows.div_ceil(32), segs],
        )?;
        if segs > 1 {
            let mconsts = [
                ("M", rows as f64),
                ("N", n as f64),
                ("Y_STRIDE", y_stride as f64),
                ("Y_OFF", y_off as f64),
                ("SEGS", segs as f64),
                ("ACC", acc as u32 as f64),
            ];
            let bufs = [
                Binding::Slice(&self.gemm_partial, 0, (segs * rows * y_stride) as u64 * 4),
                yslice,
            ];
            Self::set(
                &self.kernels,
                pass,
                name::MERGE_GEMM,
                &mconsts,
                &bufs,
                [n.div_ceil(256), 1, 1],
            )?;
        }
        Ok(())
    }

    pub fn gemv(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
    ) -> Result<()> {
        self.gemv_acc(pass, x, w, y, false)
    }

    pub fn gemv_acc(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        w: &Weight,
        y: Binding<'_>,
        acc: bool,
    ) -> Result<()> {
        let (n, k, wb, dtype) = Self::weight_io(w);
        let kb = k / 16;
        let base = n.div_ceil(256);
        let segs: u32 = [8u32, 4, 2, 1]
            .into_iter()
            .find(|s| kb % *s == 0 && base * *s >= 96)
            .unwrap_or(1);
        if segs > 1 {
            self.ensure_gemv_partial(n * segs);
        }
        let scale = Self::scale_binding(&self.dummy_scale, w);
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
            &self.kernels,
            pass,
            name::GEMV,
            &consts,
            &bufs,
            [n.div_ceil(256), segs, 1],
        )?;
        if segs > 1 {
            let bufs = [
                Binding::Slice(&self.gemv_partial, 0, n as u64 * 4 * segs as u64),
                y,
            ];
            Self::set(
                &self.kernels,
                pass,
                name::MERGE_GEMV,
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

    pub fn gemv_qkv(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        wq: &Weight,
        wk: &Weight,
        wv: &Weight,
        yq: Binding<'_>,
        yk: Binding<'_>,
        yv: Binding<'_>,
        nq: u32,
        nk: u32,
        nv: u32,
        k: u32,
    ) -> Result<()> {
        assert!(
            wq.scale().is_some() && wk.scale().is_some() && wv.scale().is_some(),
            "gemv_qkv requires quantized weights"
        );
        let ntot = nq + nk + nv;
        let kb = k / 16;
        let base = ntot.div_ceil(256);
        let segs: u32 = [16u32, 8, 4, 2, 1]
            .into_iter()
            .find(|s| kb.is_multiple_of(*s) && base * *s >= 256)
            .unwrap_or(1);
        if segs > 1 {
            self.ensure_gemv_partial(ntot * segs);
        }
        let (sq, sk, sv) = (
            Self::scale_binding(&self.dummy_scale, wq),
            Self::scale_binding(&self.dummy_scale, wk),
            Self::scale_binding(&self.dummy_scale, wv),
        );
        let out = if segs == 1 {
            yq
        } else {
            Binding::Slice(&self.gemv_partial, 0, ntot as u64 * 4 * segs as u64)
        };
        let bufs = [
            x,
            Binding::Full(wq.tensor()),
            sq,
            yq,
            Binding::Full(wk.tensor()),
            sk,
            yk,
            Binding::Full(wv.tensor()),
            sv,
            yv,
            out,
        ];
        Self::set(
            &self.kernels,
            pass,
            name::GEMV_QKV,
            &[
                ("NQ", nq as f64),
                ("NK", nk as f64),
                ("NV", nv as f64),
                ("K", k as f64),
                ("GROUP", group_const(wq)),
                ("SEGS", segs as f64),
            ],
            &bufs,
            [ntot.div_ceil(256), segs, 1],
        )?;
        if segs > 1 {
            let bufs = [
                Binding::Slice(&self.gemv_partial, 0, ntot as u64 * 4 * segs as u64),
                yq,
                yk,
                yv,
            ];
            Self::set(
                &self.kernels,
                pass,
                name::MERGE_QKV,
                &[
                    ("NQ", nq as f64),
                    ("NK", nk as f64),
                    ("NV", nv as f64),
                    ("SEGS", segs as f64),
                ],
                &bufs,
                [ntot.div_ceil(256), 1, 1],
            )?;
        }
        Ok(())
    }

    pub fn gemv_gateup(
        &mut self,
        pass: &mut Pass<'_>,
        x: Binding<'_>,
        wg: &Weight,
        wu: &Weight,
        yg: Binding<'_>,
        yu: Binding<'_>,
        n: u32,
        k: u32,
    ) -> Result<()> {
        assert!(
            wg.scale().is_some() && wu.scale().is_some(),
            "gemv_gateup requires quantized weights"
        );
        let ntot = 2 * n;
        let kb = k / 16;
        let base = ntot.div_ceil(256);
        let segs: u32 = [4u32, 2, 1]
            .into_iter()
            .find(|s| kb.is_multiple_of(*s) && base * *s >= 192)
            .unwrap_or(1);
        if segs > 1 {
            self.ensure_gemv_partial(ntot * segs);
        }
        let (sg, su) = (
            Self::scale_binding(&self.dummy_scale, wg),
            Self::scale_binding(&self.dummy_scale, wu),
        );
        let out = if segs == 1 {
            yg
        } else {
            Binding::Slice(&self.gemv_partial, 0, ntot as u64 * 4 * segs as u64)
        };
        let bufs = [
            x,
            Binding::Full(wg.tensor()),
            sg,
            yg,
            Binding::Full(wu.tensor()),
            su,
            yu,
            out,
        ];
        Self::set(
            &self.kernels,
            pass,
            name::GEMV_GATEUP,
            &[
                ("NG", n as f64),
                ("K", k as f64),
                ("GROUP", group_const(wg)),
                ("SEGS", segs as f64),
            ],
            &bufs,
            [ntot.div_ceil(256), segs, 1],
        )?;
        if segs > 1 {
            let bufs = [
                Binding::Slice(&self.gemv_partial, 0, ntot as u64 * 4 * segs as u64),
                yg,
                yu,
            ];
            Self::set(
                &self.kernels,
                pass,
                name::MERGE_GATEUP,
                &[("NG", n as f64), ("SEGS", segs as f64)],
                &bufs,
                [ntot.div_ceil(256), 1, 1],
            )?;
        }
        Ok(())
    }

    fn scale_binding<'a>(dummy: &'a Tensor, w: &'a Weight) -> Binding<'a> {
        match w.scale() {
            Some(s) => Binding::Full(s),
            None => Binding::Full(dummy),
        }
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

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return if mant == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7e00
        };
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let m = mant | 0x800000;
        let shift = (14 - half_exp) as u32;
        let shifted = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let round =
            rem > (1u32 << (shift - 1)) || (rem == (1u32 << (shift - 1)) && (shifted & 1) == 1);
        return sign | (shifted as u16).wrapping_add(round as u16);
    }
    let frac = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    let mut result = sign | ((half_exp as u16) << 10) | frac;
    if rem > 0x1000 || (rem == 0x1000 && (frac & 1) == 1) {
        result = result.wrapping_add(1);
    }
    result
}

pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

pub fn f32_to_bf16(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bits(v: f32) -> u32 {
        v.to_bits()
    }

    #[test]
    fn f16_to_f32_exact() {
        assert_eq!(f32_bits(f16_to_f32(0x0000)), f32_bits(0.0));
        assert_eq!(f32_bits(f16_to_f32(0x8000)), f32_bits(-0.0));
        assert_eq!(f32_bits(f16_to_f32(0x3C00)), f32_bits(1.0));
        assert_eq!(f32_bits(f16_to_f32(0xBC00)), f32_bits(-1.0));
        assert_eq!(f32_bits(f16_to_f32(0x7BFF)), f32_bits(65504.0));
        assert_eq!(f32_bits(f16_to_f32(0x7C00)), f32_bits(f32::INFINITY));
        assert_eq!(f32_bits(f16_to_f32(0xFC00)), f32_bits(f32::NEG_INFINITY));
        assert_eq!(f32_bits(f16_to_f32(0x0001)), f32_bits(2.0f32.powi(-24)));
    }

    #[test]
    fn f32_to_f16_exact() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(-1.0), 0xBC00);
        assert_eq!(f32_to_f16(1.5), 0x3E00);
        assert_eq!(f32_to_f16(2.0), 0x4000);
        assert_eq!(f32_to_f16(-2.0), 0xC000);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f32_to_f16(65504.0), 0x7BFF);
    }

    #[test]
    fn f32_to_f16_rounds_nearest_even() {
        assert_eq!(f32_to_f16(1.0 + 2.0_f32.powi(-11)), 0x3C00);
        assert_eq!(f32_to_f16(1.0 + 2.0_f32.powi(-10)), 0x3C01);
        assert_eq!(f32_to_f16(0.1), 0x2E66);
    }

    #[test]
    fn f32_to_f16_saturates() {
        assert_eq!(f32_to_f16(65520.0), 0x7C00);
        assert_eq!(f32_to_f16(1e30), 0x7C00);
        assert_eq!(f32_to_f16(-1e30), 0xFC00);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7C00);
        assert_eq!(f32_to_f16(f32::NAN), 0x7E00);
    }

    #[test]
    fn f32_to_f16_flushes_denormals() {
        assert_eq!(f32_to_f16(2.0_f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_f16(2.0_f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_f16(-2.0_f32.powi(-24)), 0x8001);
        assert_eq!(f32_to_f16(1.5 * 2.0_f32.powi(-24)), 0x0002);
    }

    #[test]
    fn f32_to_f16_round_trips() {
        for v in [0.0, 1.0, -1.0, 0.5, 65504.0, 1.5, 2.0, -2.0] {
            assert_eq!(f16_to_f32(f32_to_f16(v)), v);
        }
    }

    #[test]
    fn bf16_round_trips() {
        for v in [0.0, 1.0, -1.0, 1.5, 123.0, -0.25, 65536.0] {
            assert_eq!(bf16_to_f32(f32_to_bf16(v)), v);
        }
    }
}

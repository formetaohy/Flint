//! Narrow floating-point conversions shared by every format decoder: f16 and
//! bf16 payloads arrive as raw little-endian bytes and widen to f32 here;
//! f32 narrows back for writers.

/// IEEE 754 half → single precision, subnormals included.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal half: normalize into a normal float.
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

/// IEEE 754 single → half precision, round-to-nearest-even.
pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;

    if exp == 0xff {
        // Inf/NaN: keep the payload's top 10 bits.
        return sign | 0x7c00 | (mant >> 13) as u16;
    }
    let (exp16, mant16) = if exp > 0x8e {
        // Overflow to infinity (or keep max finite with round-to-nearest).
        if exp > 0x8f || (exp == 0x8f && mant != 0) {
            (0x1f, 0u32)
        } else {
            (0x1e, 0x3ff)
        }
    } else if exp <= 0x70 {
        // Subnormal or zero: scale the mantissa into half's subnormal range.
        let m = mant | 0x80_0000;
        let shift = 0x71 - exp;
        if shift >= 24 {
            (0, 0)
        } else {
            let m16 = m >> shift;
            let round = (m >> (shift - 1)) & 1;
            (0, m16 + round)
        }
    } else {
        let e16 = (exp - 112) as u32;
        let m16 = mant >> 13;
        let round = (mant >> 12) & 1;
        let carry = m16 + round;
        if carry > 0x3ff {
            (e16 + 1, 0)
        } else {
            (e16, carry)
        }
    };
    sign | ((exp16 as u16) << 10) | mant16 as u16
}

/// bfloat16 → single precision: the top 16 bits of the f32 bit pattern.
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Single → bfloat16: truncation of the f32 bit pattern (round-to-nearest is
/// not required by any consuming format).
pub fn f32_to_bf16(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

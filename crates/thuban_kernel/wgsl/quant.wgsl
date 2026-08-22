// Native ggml block decode, shared by the gemm/gemv kernels.
// `w` holds block bytes with every block start padded to 4-byte alignment
// (see thuban_tensor::quant), `lut` holds the grid tables
// (see thuban_tensor::quant::lut_bytes).

const LUT_IQ2XXS: u32 = 0u;
const LUT_IQ2XS: u32 = 2048u;
const LUT_IQ3XXS: u32 = 6144u;
const LUT_IQ1S: u32 = 7168u;
const LUT_IQ3S: u32 = 23552u;
const LUT_IQ2S: u32 = 25600u;
const LUT_IQ4NL: u32 = 33792u;
const LUT_KSIGNS: u32 = 33808u;
const LUT_KMASK: u32 = 33936u;

fn qword(b: u32) -> u32 {
    return w[b / 4u];
}

fn qbyte(b: u32) -> u32 {
    let v = w[b / 4u];
    return (v >> (8u * (b % 4u))) & 255u;
}

fn qu16(b: u32) -> u32 {
    let v = w[b / 4u];
    if b % 4u == 0u {
        return v & 65535u;
    }
    return v >> 16u;
}

fn qu32(b: u32) -> u32 {
    let v = w[b / 4u];
    let sh = 8u * (b % 4u);
    if sh == 0u {
        return v;
    }
    return (v >> sh) | (w[b / 4u + 1u] << (32u - sh));
}

fn lbyte(b: u32) -> u32 {
    let v = lut[b / 4u];
    return (v >> (8u * (b % 4u))) & 255u;
}

fn li8(b: u32) -> i32 {
    let v = lbyte(b);
    return select(i32(v), i32(v) - 256, v >= 128u);
}

fn to_i8(v: u32) -> i32 {
    return select(i32(v & 255u), i32(v & 255u) - 256, (v & 255u) >= 128u);
}

fn li8x4(b: u32) -> vec4<f32> {
    let v = lut[b / 4u];
    return vec4<f32>(
        f32(to_i8(v)),
        f32(to_i8(v >> 8u)),
        f32(to_i8(v >> 16u)),
        f32(to_i8(v >> 24u)),
    );
}

fn qi8(v: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(to_i8(v)),
        f32(to_i8(v >> 8u)),
        f32(to_i8(v >> 16u)),
        f32(to_i8(v >> 24u)),
    );
}

fn nib4(v: u32, s: u32) -> vec4<f32> {
    return vec4<f32>(
        f32((v >> s) & 15u),
        f32((v >> (s + 8u)) & 15u),
        f32((v >> (s + 16u)) & 15u),
        f32((v >> (s + 24u)) & 15u),
    );
}

fn widen16(h: u32) -> f32 {
    let sign = (h & 32768u) << 16u;
    let exp = (h >> 10u) & 31u;
    let man = h & 1023u;
    if exp == 31u {
        return select(65504.0, 1.0e30, man == 0u);
    }
    if exp == 0u && man == 0u {
        return bitcast<f32>(sign);
    }
    var e = i32(exp) + 112;
    var m = man;
    if exp == 0u {
        e = 113;
        while (m & 1024u) == 0u {
            m = m << 1u;
            e = e - 1;
        }
    }
    let bits = sign | (u32(e) << 23u) | ((m & 1023u) << 13u);
    return bitcast<f32>(bits);
}

fn qf16(b: u32) -> f32 {
    return widen16(qu16(b));
}

fn wrow(ty: u32, K: u32) -> u32 {
    if ty == 0u {
        return K * 4u;
    }
    if ty == 1u || ty == 30u {
        return K * 2u;
    }
    if ty == 2u || ty == 3u || ty == 20u {
        return (K / 32u) * 20u;
    }
    if ty == 6u || ty == 7u {
        return (K / 32u) * 24u;
    }
    if ty == 8u {
        return (K / 32u) * 36u;
    }
    if ty == 10u {
        return (K / 256u) * 84u;
    }
    if ty == 11u || ty == 21u {
        return (K / 256u) * 112u;
    }
    if ty == 12u {
        return (K / 256u) * 144u;
    }
    if ty == 13u {
        return (K / 256u) * 176u;
    }
    if ty == 14u {
        return (K / 256u) * 212u;
    }
    if ty == 15u {
        return (K / 256u) * 292u;
    }
    if ty == 16u {
        return (K / 256u) * 68u;
    }
    if ty == 17u {
        return (K / 256u) * 76u;
    }
    if ty == 18u {
        return (K / 256u) * 100u;
    }
    if ty == 19u {
        return (K / 256u) * 52u;
    }
    if ty == 22u {
        return (K / 256u) * 84u;
    }
    if ty == 23u {
        return (K / 256u) * 136u;
    }
    if ty == 29u {
        return (K / 256u) * 56u;
    }
    if ty == 34u {
        return (K / 256u) * 56u;
    }
    if ty == 35u {
        return (K / 256u) * 68u;
    }
    return 0u;
}

fn scale_min_k4(p: u32, sb: u32) -> vec2<f32> {
    if p < 4u {
        return vec2<f32>(f32(qbyte(sb + p) & 63u), f32(qbyte(sb + p + 4u) & 63u));
    }
    let dsc = (qbyte(sb + p + 4u) & 15u) | ((qbyte(sb + p - 4u) >> 6u) << 4u);
    let msc = (qbyte(sb + p + 4u) >> 4u) | ((qbyte(sb + p) >> 6u) << 4u);
    return vec2<f32>(f32(dsc), f32(msc));
}

fn q3k_scale(b: u32, idx: u32) -> f32 {
    let aux0 = qword(b);
    let aux1 = qword(b + 4u);
    let tmp = qword(b + 8u);
    let a2 = ((aux0 >> 4u) & 252645135u) | (((tmp >> 4u) & 50529027u) << 4u);
    let a3 = ((aux1 >> 4u) & 252645135u) | (((tmp >> 6u) & 50529027u) << 4u);
    let a0 = (aux0 & 252645135u) | ((tmp & 50529027u) << 4u);
    let a1 = (aux1 & 252645135u) | (((tmp >> 2u) & 50529027u) << 4u);
    let aux = select(select(a0, a1, (idx / 4u) == 1u), select(a2, a3, (idx / 4u) == 3u), (idx / 4u) >= 2u);
    return f32(to_i8(aux >> (8u * (idx % 4u))));
}

// Decodes the 32-element tile of row n at column kb (kb % 32 == 0) into out.
fn tile32(ty: u32, n: u32, kb: u32, K: u32, off: u32, out: ptr<function, array<vec4<f32>, 8>>) {
    let nb = n * wrow(ty, K) + off;
    switch ty {
        case 0u: {
            let b = nb / 4u + kb;
            for (var q = 0u; q < 8u; q++) {
                let o = b + 4u * q;
                (*out)[q] = vec4<f32>(
                    bitcast<f32>(w[o]),
                    bitcast<f32>(w[o + 1u]),
                    bitcast<f32>(w[o + 2u]),
                    bitcast<f32>(w[o + 3u]),
                );
            }
        }
        case 1u: {
            let b = nb / 4u + kb / 2u;
            for (var q = 0u; q < 8u; q++) {
                let o = b + 2u * q;
                let v0 = w[o];
                let v1 = w[o + 1u];
                (*out)[q] = vec4<f32>(
                    widen16(v0 & 65535u),
                    widen16(v0 >> 16u),
                    widen16(v1 & 65535u),
                    widen16(v1 >> 16u),
                );
            }
        }
        case 30u: {
            let b = nb / 4u + kb / 2u;
            for (var q = 0u; q < 8u; q++) {
                let o = b + 2u * q;
                let v0 = w[o];
                let v1 = w[o + 1u];
                (*out)[q] = vec4<f32>(
                    bitcast<f32>((v0 & 65535u) << 16u),
                    bitcast<f32>((v0 >> 16u) << 16u),
                    bitcast<f32>((v1 & 65535u) << 16u),
                    bitcast<f32>((v1 >> 16u) << 16u),
                );
            }
        }
        case 2u: {
            let b = nb + (kb / 32u) * 20u;
            let d = qf16(b);
            for (var q = 0u; q < 4u; q++) {
                let v = qu32(b + 2u + 4u * q);
                (*out)[q] = (nib4(v, 0u) - 8.0) * d;
                (*out)[q + 4u] = (nib4(v, 4u) - 8.0) * d;
            }
        }
        case 3u: {
            let b = nb + (kb / 32u) * 20u;
            let d = qf16(b);
            let m = qf16(b + 2u);
            for (var q = 0u; q < 4u; q++) {
                let v = qu32(b + 4u + 4u * q);
                (*out)[q] = nib4(v, 0u) * d + m;
                (*out)[q + 4u] = nib4(v, 4u) * d + m;
            }
        }
        case 6u: {
            let b = nb + (kb / 32u) * 24u;
            let d = qf16(b);
            let qh = qu32(b + 2u);
            for (var q = 0u; q < 4u; q++) {
                let v = qu32(b + 6u + 4u * q);
                let hi = (qh >> (4u * q)) & 15u;
                let hi2 = (qh >> (16u + 4u * q)) & 15u;
                let qlo = ((v & 15u) | ((hi & 1u) << 4u))
                    | (((v >> 8u) & 15u) | (((hi >> 1u) & 1u) << 4u)) << 8u
                    | (((v >> 16u) & 15u) | (((hi >> 2u) & 1u) << 4u)) << 16u
                    | (((v >> 24u) & 15u) | (((hi >> 3u) & 1u) << 4u)) << 24u;
                let qhi = (((v >> 4u) & 15u) | ((hi2 & 1u) << 4u))
                    | (((v >> 12u) & 15u) | (((hi2 >> 1u) & 1u) << 4u)) << 8u
                    | (((v >> 20u) & 15u) | (((hi2 >> 2u) & 1u) << 4u)) << 16u
                    | (((v >> 28u) & 15u) | (((hi2 >> 3u) & 1u) << 4u)) << 24u;
                (*out)[q] = (qi8(qlo) - 16.0) * d;
                (*out)[q + 4u] = (qi8(qhi) - 16.0) * d;
            }
        }
        case 7u: {
            let b = nb + (kb / 32u) * 24u;
            let d = qf16(b);
            let m = qf16(b + 2u);
            let qh = qu32(b + 4u);
            for (var q = 0u; q < 4u; q++) {
                let v = qu32(b + 8u + 4u * q);
                let hi = (qh >> (4u * q)) & 15u;
                let hi2 = (qh >> (16u + 4u * q)) & 15u;
                let qlo = ((v & 15u) | ((hi & 1u) << 4u))
                    | (((v >> 8u) & 15u) | (((hi >> 1u) & 1u) << 4u)) << 8u
                    | (((v >> 16u) & 15u) | (((hi >> 2u) & 1u) << 4u)) << 16u
                    | (((v >> 24u) & 15u) | (((hi >> 3u) & 1u) << 4u)) << 24u;
                let qhi = (((v >> 4u) & 15u) | ((hi2 & 1u) << 4u))
                    | (((v >> 12u) & 15u) | (((hi2 >> 1u) & 1u) << 4u)) << 8u
                    | (((v >> 20u) & 15u) | (((hi2 >> 2u) & 1u) << 4u)) << 16u
                    | (((v >> 28u) & 15u) | (((hi2 >> 3u) & 1u) << 4u)) << 24u;
                (*out)[q] = qi8(qlo) * d + m;
                (*out)[q + 4u] = qi8(qhi) * d + m;
            }
        }
        case 8u: {
            let b = nb + (kb / 32u) * 36u;
            let d = qf16(b);
            for (var q = 0u; q < 8u; q++) {
                (*out)[q] = qi8(qu32(b + 2u + 4u * q)) * d;
            }
        }
        case 10u: {
            let b = nb + (kb / 256u) * 84u;
            let d = qf16(b + 80u);
            let m = qf16(b + 82u);
            let s = (kb % 256u) / 32u;
            let j = s % 4u;
            let b2 = 2u * j;
            let qo = b + 16u + 32u * (s / 4u);
            for (var h = 0u; h < 2u; h++) {
                let sc = qbyte(b + 2u * s + h);
                let dl = d * f32(sc & 15u);
                let ml = m * f32(sc >> 4u);
                for (var q = 0u; q < 4u; q++) {
                    let v = qword(qo + 16u * h + 4u * q);
                    (*out)[4u * h + q] = (vec4<f32>(
                        f32((v >> b2) & 3u),
                        f32((v >> (b2 + 8u)) & 3u),
                        f32((v >> (b2 + 16u)) & 3u),
                        f32((v >> (b2 + 24u)) & 3u),
                    )) * dl - ml;
                }
            }
        }
        case 11u: {
            let b = nb + (kb / 256u) * 112u;
            let d = qf16(b + 108u);
            let s = (kb % 256u) / 32u;
            let j = s % 4u;
            let b2 = 2u * j;
            let qo = b + 32u + 32u * (s / 4u);
            for (var h = 0u; h < 2u; h++) {
                let dl = d * (q3k_scale(b + 96u, 2u * s + h) - 32.0);
                for (var q = 0u; q < 4u; q++) {
                    let v = qword(qo + 16u * h + 4u * q);
                    let hm = qword(b + 16u * h + 4u * q);
                    let q4 = vec4<f32>(
                        f32((v >> b2) & 3u),
                        f32((v >> (b2 + 8u)) & 3u),
                        f32((v >> (b2 + 16u)) & 3u),
                        f32((v >> (b2 + 24u)) & 3u),
                    );
                    let m4 = vec4<f32>(
                        f32(select(0, -4, ((hm >> s) & 1u) == 0u)),
                        f32(select(0, -4, ((hm >> (s + 8u)) & 1u) == 0u)),
                        f32(select(0, -4, ((hm >> (s + 16u)) & 1u) == 0u)),
                        f32(select(0, -4, ((hm >> (s + 24u)) & 1u) == 0u)),
                    );
                    (*out)[4u * h + q] = (q4 + m4) * dl;
                }
            }
        }
        case 12u: {
            let b = nb + (kb / 256u) * 144u;
            let d = qf16(b);
            let m = qf16(b + 2u);
            let p = (kb % 256u) / 32u;
            let sm = scale_min_k4(p, b + 4u);
            let dl = d * sm.x;
            let ml = m * sm.y;
            let sh = 4u * (p % 2u);
            let qo = b + 16u + (p / 2u) * 32u;
            for (var q = 0u; q < 8u; q++) {
                (*out)[q] = nib4(qword(qo + 4u * q), sh) * dl - ml;
            }
        }
        case 13u: {
            let b = nb + (kb / 256u) * 176u;
            let d = qf16(b);
            let m = qf16(b + 2u);
            let p = (kb % 256u) / 32u;
            let sm = scale_min_k4(p, b + 4u);
            let dl = d * sm.x;
            let ml = m * sm.y;
            let sh = 4u * (p % 2u);
            let qo = b + 48u + (p / 2u) * 32u;
            for (var q = 0u; q < 8u; q++) {
                let v = qword(qo + 4u * q);
                let hb = (qword(b + 16u + 4u * q) >> p) & 16843009u;
                let q4 = nib4(v, sh) + vec4<f32>(
                    f32((hb & 1u) << 4u),
                    f32(((hb >> 8u) & 1u) << 4u),
                    f32(((hb >> 16u) & 1u) << 4u),
                    f32(((hb >> 24u) & 1u) << 4u),
                );
                (*out)[q] = q4 * dl - ml;
            }
        }
        case 14u: {
            let b = nb + (kb / 256u) * 212u;
            let d = qf16(b + 208u);
            let s = (kb % 256u) / 32u;
            let n2 = s / 4u;
            let t = s % 4u;
            let hb = 2u * t;
            let nb2 = select(0u, 4u, t >= 2u);
            let qlo = b + n2 * 64u + (t % 2u) * 32u;
            let qho = b + 128u + n2 * 32u;
            for (var h = 0u; h < 2u; h++) {
                let sc = d * f32(to_i8(qbyte(b + 192u + 8u * n2 + 2u * t + h)));
                for (var q = 0u; q < 4u; q++) {
                    let i = 16u * h + 4u * q;
                    let ql = qword(qlo + i);
                    let qh = qword(qho + i);
                    let q4 = vec4<f32>(
                        f32(((ql >> nb2) & 15u) | (((qh >> hb) & 3u) << 4u)),
                        f32(((ql >> (nb2 + 8u)) & 15u) | (((qh >> (hb + 8u)) & 3u) << 4u)),
                        f32(((ql >> (nb2 + 16u)) & 15u) | (((qh >> (hb + 16u)) & 3u) << 4u)),
                        f32(((ql >> (nb2 + 24u)) & 15u) | (((qh >> (hb + 24u)) & 3u) << 4u)),
                    );
                    (*out)[4u * h + q] = (q4 - 32.0) * sc;
                }
            }
        }
        case 15u: {
            let b = nb + (kb / 256u) * 292u;
            let d = bitcast<f32>(qword(b));
            let i = kb % 256u;
            for (var q = 0u; q < 8u; q++) {
                (*out)[q] = qi8(qword(b + 4u + i + 4u * q)) * d;
            }
        }
        case 16u: {
            let b = nb + (kb / 256u) * 68u;
            let d = qf16(b);
            let ib32 = (kb % 256u) / 32u;
            let aux0 = qu32(b + 2u + 8u * ib32);
            let aux1 = qu32(b + 6u + 8u * ib32);
            let db = d * (0.5 + f32(aux1 >> 28u)) * 0.25;
            for (var l = 0u; l < 4u; l++) {
                let idx = ((aux0 >> (8u * l)) & 255u) * 8u;
                let g = li8x4(LUT_IQ2XXS + idx) * db;
                let g2 = li8x4(LUT_IQ2XXS + idx + 4u) * db;
                let signs = lbyte(LUT_KSIGNS + ((aux1 >> (7u * l)) & 127u));
                let s4 = vec4<f32>(
                    select(1.0, -1.0, (signs & 1u) != 0u),
                    select(1.0, -1.0, (signs & 2u) != 0u),
                    select(1.0, -1.0, (signs & 4u) != 0u),
                    select(1.0, -1.0, (signs & 8u) != 0u),
                );
                let s4h = vec4<f32>(
                    select(1.0, -1.0, (signs & 16u) != 0u),
                    select(1.0, -1.0, (signs & 32u) != 0u),
                    select(1.0, -1.0, (signs & 64u) != 0u),
                    select(1.0, -1.0, (signs & 128u) != 0u),
                );
                (*out)[2u * l] = g * s4;
                (*out)[2u * l + 1u] = g2 * s4h;
            }
        }
        case 17u: {
            let b = nb + (kb / 256u) * 76u;
            let d = qf16(b);
            let ib32 = (kb % 256u) / 32u;
            let sc = qbyte(b + 66u + ib32);
            let db0 = d * (0.5 + f32(sc & 15u)) * 0.25;
            let db1 = d * (0.5 + f32(sc >> 4u)) * 0.25;
            for (var l = 0u; l < 4u; l++) {
                let q = qu16(b + 2u + 2u * (4u * ib32 + l));
                let idx = (q & 511u) * 8u;
                let db = select(db0, db1, (l / 2u) == 1u);
                let g = li8x4(LUT_IQ2XS + idx) * db;
                let g2 = li8x4(LUT_IQ2XS + idx + 4u) * db;
                let signs = lbyte(LUT_KSIGNS + (q >> 9u));
                let s4 = vec4<f32>(
                    select(1.0, -1.0, (signs & 1u) != 0u),
                    select(1.0, -1.0, (signs & 2u) != 0u),
                    select(1.0, -1.0, (signs & 4u) != 0u),
                    select(1.0, -1.0, (signs & 8u) != 0u),
                );
                let s4h = vec4<f32>(
                    select(1.0, -1.0, (signs & 16u) != 0u),
                    select(1.0, -1.0, (signs & 32u) != 0u),
                    select(1.0, -1.0, (signs & 64u) != 0u),
                    select(1.0, -1.0, (signs & 128u) != 0u),
                );
                (*out)[2u * l] = g * s4;
                (*out)[2u * l + 1u] = g2 * s4h;
            }
        }
        case 18u: {
            let b = nb + (kb / 256u) * 100u;
            let d = qf16(b);
            let ib32 = (kb % 256u) / 32u;
            let aux = qu32(b + 66u + 4u * ib32);
            let db = d * (0.5 + f32(aux >> 28u)) * 0.5;
            for (var l = 0u; l < 4u; l++) {
                let signs = lbyte(LUT_KSIGNS + ((aux >> (7u * l)) & 127u));
                let s4 = vec4<f32>(
                    select(1.0, -1.0, (signs & 1u) != 0u),
                    select(1.0, -1.0, (signs & 2u) != 0u),
                    select(1.0, -1.0, (signs & 4u) != 0u),
                    select(1.0, -1.0, (signs & 8u) != 0u),
                );
                let s4h = vec4<f32>(
                    select(1.0, -1.0, (signs & 16u) != 0u),
                    select(1.0, -1.0, (signs & 32u) != 0u),
                    select(1.0, -1.0, (signs & 64u) != 0u),
                    select(1.0, -1.0, (signs & 128u) != 0u),
                );
                let i0 = qbyte(b + 2u + 2u * (4u * ib32 + l));
                let i1 = qbyte(b + 3u + 2u * (4u * ib32 + l));
                (*out)[2u * l] = li8x4(LUT_IQ3XXS + i0 * 4u) * (db * s4);
                (*out)[2u * l + 1u] = li8x4(LUT_IQ3XXS + i1 * 4u) * (db * s4h);
            }
        }
        case 19u: {
            let b = nb + (kb / 256u) * 52u;
            let d = qf16(b);
            let ib = (kb % 256u) / 32u;
            let qh = qu16(b + 34u + 2u * ib);
            let dl = d * (2.0 * f32((qh >> 12u) & 7u) + 1.0);
            let delta = select(0.125, -0.125, (qh & 32768u) != 0u);
            for (var l = 0u; l < 4u; l++) {
                let idx = (qbyte(b + 2u + 4u * ib + l) | (((qh >> (3u * l)) & 7u) << 8u)) * 8u;
                (*out)[2u * l] = (li8x4(LUT_IQ1S + idx) + delta) * dl;
                (*out)[2u * l + 1u] = (li8x4(LUT_IQ1S + idx + 4u) + delta) * dl;
            }
        }
        case 20u: {
            let b = nb + (kb / 32u) * 20u;
            let d = qf16(b);
            for (var q = 0u; q < 4u; q++) {
                let v = qu32(b + 2u + 4u * q);
                let v0 = li8(LUT_IQ4NL + (v & 15u));
                let v1 = li8(LUT_IQ4NL + ((v >> 8u) & 15u));
                let v2 = li8(LUT_IQ4NL + ((v >> 16u) & 15u));
                let v3 = li8(LUT_IQ4NL + ((v >> 24u) & 15u));
                let v4 = li8(LUT_IQ4NL + ((v >> 4u) & 15u));
                let v5 = li8(LUT_IQ4NL + ((v >> 12u) & 15u));
                let v6 = li8(LUT_IQ4NL + ((v >> 20u) & 15u));
                let v7 = li8(LUT_IQ4NL + ((v >> 28u) & 15u));
                (*out)[q] = vec4<f32>(f32(v0), f32(v1), f32(v2), f32(v3)) * d;
                (*out)[q + 4u] = vec4<f32>(f32(v4), f32(v5), f32(v6), f32(v7)) * d;
            }
        }
        case 21u: {
            let b = nb + (kb / 256u) * 112u;
            let d = qf16(b);
            let ib32 = (kb % 256u) / 32u;
            let scb = qbyte(b + 106u + ib32 / 2u);
            let db = d * (1.0 + 2.0 * f32(select(scb & 15u, scb >> 4u, (ib32 % 2u) == 1u)));
            let qh = qbyte(b + 66u + ib32);
            for (var l = 0u; l < 4u; l++) {
                let sh = 8u - 2u * l;
                let i0 = qbyte(b + 2u + 8u * ib32 + 2u * l) | ((qh << sh) & 256u);
                let i1 = qbyte(b + 3u + 8u * ib32 + 2u * l) | ((qh << (sh - 1u)) & 256u);
                let signs = qbyte(b + 74u + 4u * ib32 + l);
                let s4 = vec4<f32>(
                    select(1.0, -1.0, (signs & 1u) != 0u),
                    select(1.0, -1.0, (signs & 2u) != 0u),
                    select(1.0, -1.0, (signs & 4u) != 0u),
                    select(1.0, -1.0, (signs & 8u) != 0u),
                );
                let s4h = vec4<f32>(
                    select(1.0, -1.0, (signs & 16u) != 0u),
                    select(1.0, -1.0, (signs & 32u) != 0u),
                    select(1.0, -1.0, (signs & 64u) != 0u),
                    select(1.0, -1.0, (signs & 128u) != 0u),
                );
                (*out)[2u * l] = li8x4(LUT_IQ3S + i0 * 4u) * (db * s4);
                (*out)[2u * l + 1u] = li8x4(LUT_IQ3S + i1 * 4u) * (db * s4h);
            }
        }
        case 22u: {
            let b = nb + (kb / 256u) * 84u;
            let d = qf16(b);
            let ib32 = (kb % 256u) / 32u;
            let sc = qbyte(b + 74u + ib32);
            let db0 = d * (0.5 + f32(sc & 15u)) * 0.25;
            let db1 = d * (0.5 + f32(sc >> 4u)) * 0.25;
            let qh = qbyte(b + 66u + ib32);
            for (var l = 0u; l < 4u; l++) {
                let idx = (qbyte(b + 2u + 4u * ib32 + l) | ((qh << (8u - 2u * l)) & 768u)) * 8u;
                let db = select(db0, db1, (l / 2u) == 1u);
                let signs = qbyte(b + 34u + 4u * ib32 + l);
                let s4 = vec4<f32>(
                    select(1.0, -1.0, (signs & 1u) != 0u),
                    select(1.0, -1.0, (signs & 2u) != 0u),
                    select(1.0, -1.0, (signs & 4u) != 0u),
                    select(1.0, -1.0, (signs & 8u) != 0u),
                );
                let s4h = vec4<f32>(
                    select(1.0, -1.0, (signs & 16u) != 0u),
                    select(1.0, -1.0, (signs & 32u) != 0u),
                    select(1.0, -1.0, (signs & 64u) != 0u),
                    select(1.0, -1.0, (signs & 128u) != 0u),
                );
                (*out)[2u * l] = li8x4(LUT_IQ2S + idx) * (db * s4);
                (*out)[2u * l + 1u] = li8x4(LUT_IQ2S + idx + 4u) * (db * s4h);
            }
        }
        case 23u: {
            let b = nb + (kb / 256u) * 136u;
            let d = qf16(b);
            let ib32 = (kb % 256u) / 32u;
            let ls = ((qbyte(b + 4u + ib32 / 2u) >> (4u * (ib32 % 2u))) & 15u)
                | (((qu16(b + 2u) >> (2u * ib32)) & 3u) << 4u);
            let dl = d * (f32(ls) - 32.0);
            let qo = b + 8u + ib32 * 16u;
            for (var q = 0u; q < 4u; q++) {
                let v = qword(qo + 4u * q);
                let v0 = li8(LUT_IQ4NL + (v & 15u));
                let v1 = li8(LUT_IQ4NL + ((v >> 8u) & 15u));
                let v2 = li8(LUT_IQ4NL + ((v >> 16u) & 15u));
                let v3 = li8(LUT_IQ4NL + ((v >> 24u) & 15u));
                let v4 = li8(LUT_IQ4NL + ((v >> 4u) & 15u));
                let v5 = li8(LUT_IQ4NL + ((v >> 12u) & 15u));
                let v6 = li8(LUT_IQ4NL + ((v >> 20u) & 15u));
                let v7 = li8(LUT_IQ4NL + ((v >> 28u) & 15u));
                (*out)[q] = vec4<f32>(f32(v0), f32(v1), f32(v2), f32(v3)) * dl;
                (*out)[q + 4u] = vec4<f32>(f32(v4), f32(v5), f32(v6), f32(v7)) * dl;
            }
        }
        case 29u: {
            let b = nb + (kb / 256u) * 56u;
            let sc0 = qu16(b + 48u);
            let sc1 = qu16(b + 50u);
            let sc2 = qu16(b + 52u);
            let sc3 = qu16(b + 54u);
            let scale = (sc0 >> 12u) | ((sc1 >> 8u) & 240u) | ((sc2 >> 4u) & 3840u) | (sc3 & 61440u);
            let d = widen16(scale);
            let ib = (kb % 256u) / 32u;
            let sc = qu16(b + 48u + 2u * (ib / 2u));
            let dl1 = d * (2.0 * f32((sc >> (6u * (ib % 2u))) & 7u) + 1.0);
            let dl2 = d * (2.0 * f32((sc >> (6u * (ib % 2u) + 3u)) & 7u) + 1.0);
            let qh0 = qbyte(b + 32u + 2u * ib);
            let qh1 = qbyte(b + 33u + 2u * ib);
            for (var l = 0u; l < 4u; l++) {
                let qhb = select(qh0, qh1, l >= 2u);
                let shift = select(8u, 4u, (l % 2u) == 1u);
                let idx = (qbyte(b + 4u * ib + l) | ((qhb << shift) & 1792u)) * 8u;
                let delta = select(0.125, -0.125, (qhb & select(8u, 128u, (l % 2u) == 1u)) != 0u);
                let dl = select(dl1, dl2, l >= 2u);
                (*out)[2u * l] = (li8x4(LUT_IQ1S + idx) + delta) * dl;
                (*out)[2u * l + 1u] = (li8x4(LUT_IQ1S + idx + 4u) + delta) * dl;
            }
        }
        case 34u: {
            let b = nb + (kb / 256u) * 56u;
            let d = qf16(b + 52u);
            for (var q = 0u; q < 8u; q++) {
                var v = vec4<f32>();
                for (var t = 0u; t < 4u; t++) {
                    let r = (kb % 256u) + 4u * q + t;
                    var byte: u32;
                    var p: u32;
                    if r < 160u {
                        byte = qbyte(b + r % 32u);
                        p = r / 32u;
                    } else if r < 240u {
                        byte = qbyte(b + 32u + (r - 160u) % 16u);
                        p = (r - 160u) / 16u;
                    } else {
                        byte = qbyte(b + 48u + (r - 240u) % 4u);
                        p = (r - 240u) / 4u;
                    }
                    var pw: u32 = 1u;
                    if p == 1u {
                        pw = 3u;
                    } else if p == 2u {
                        pw = 9u;
                    } else if p == 3u {
                        pw = 27u;
                    } else if p == 4u {
                        pw = 81u;
                    }
                    let val = (f32(((byte * pw) & 255u) * 3u >> 8u) - 1.0) * d;
                    if t == 0u {
                        v.x = val;
                    } else if t == 1u {
                        v.y = val;
                    } else if t == 2u {
                        v.z = val;
                    } else {
                        v.w = val;
                    }
                }
                (*out)[q] = v;
            }
        }
        case 35u: {
            let b = nb + (kb / 256u) * 68u;
            let d = qf16(b + 64u);
            let r = kb % 256u;
            let l = r / 32u;
            let s = 2u * (l % 4u);
            let qo = b + 32u * (l / 4u);
            for (var q = 0u; q < 8u; q++) {
                let v = qword(qo + 4u * q);
                (*out)[q] = (vec4<f32>(
                    f32((v >> s) & 3u),
                    f32((v >> (s + 8u)) & 3u),
                    f32((v >> (s + 16u)) & 3u),
                    f32((v >> (s + 24u)) & 3u),
                ) - 1.0) * d;
            }
        }
        default: {
            for (var q = 0u; q < 8u; q++) {
                (*out)[q] = vec4<f32>();
            }
        }
    }
}

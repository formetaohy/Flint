//! ggml block decoders, validated against independent reference decodes
//! written straight from the ggml block struct layouts (deliberately different
//! loop structures than the production code, so index bugs cannot match).

use flint_checkpoint::TensorData;
use flint_checkpoint::dequant::{GgmlType, f16_to_f32, to_f32};

// ---------------------------------------------------------------- f16

#[test]
fn f16_covers_the_ieee754_cases() {
    let cases: &[(u16, f32)] = &[
        (0x0000, 0.0),
        (0x8000, -0.0),
        (0x3C00, 1.0),
        (0xC000, -2.0),
        (0x4100, 2.5),
        (0x7C00, f32::INFINITY),
        (0xFC00, f32::NEG_INFINITY),
        (0x0001, 2.0f32.powi(-24)), // smallest subnormal
        (0x03FF, 2.0f32.powi(-14) * (1023.0 / 1024.0)), // largest subnormal
        (0x3555, 1.0 / 3.0),        // normal with rounding
    ];
    for &(bits, want) in cases {
        let got = f16_to_f32(bits);
        if want.is_infinite() {
            assert_eq!(got, want, "bits {bits:#06x}");
            continue;
        }
        assert!(
            (got - want).abs() <= want.abs() * 1e-3 + 1e-45,
            "bits {bits:#06x}: {got} vs {want}"
        );
    }
    assert!(f16_to_f32(0x7E00).is_nan());
}

// ---------------------------------------------------------------- type table

#[test]
fn type_table_matches_ggml() {
    let valid: &[(u32, GgmlType, usize, usize)] = &[
        (0, GgmlType::F32, 1, 4),
        (1, GgmlType::F16, 1, 2),
        (2, GgmlType::Q4_0, 32, 18),
        (3, GgmlType::Q4_1, 32, 20),
        (6, GgmlType::Q5_0, 32, 22),
        (7, GgmlType::Q5_1, 32, 24),
        (8, GgmlType::Q8_0, 32, 34),
        (10, GgmlType::Q2K, 256, 84),
        (11, GgmlType::Q3K, 256, 110),
        (12, GgmlType::Q4K, 256, 144),
        (13, GgmlType::Q5K, 256, 176),
        (14, GgmlType::Q6K, 256, 210),
        (30, GgmlType::Bf16, 1, 2),
    ];
    for &(tag, ty, bl, bb) in valid {
        assert_eq!(GgmlType::from_u32(tag).unwrap(), ty);
        assert_eq!(ty.block_len(), bl);
        assert_eq!(ty.block_bytes(), bb);
    }
    for tag in [4u32, 5, 9, 15, 29, 31] {
        assert!(
            GgmlType::from_u32(tag).is_err(),
            "tag {tag} must be rejected"
        );
    }
}

// ---------------------------------------------------------------- drivers

/// Deterministic byte filler so every block is exercised with arbitrary bits.
struct Bytes(u64);
impl Bytes {
    fn take(&mut self, n: usize) -> Vec<u8> {
        (0..n)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (self.0 >> 56) as u8
            })
            .collect()
    }
}

fn half_bits(v: f32) -> [u8; 2] {
    // Nearest-even enough for the small exact values used here.
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (b >> 13) & 0x3ff;
    let h = if exp <= 0 {
        0
    } else if exp >= 0x1f {
        0x7c00
    } else {
        sign | ((exp as u16) << 10) | mant as u16
    };
    h.to_le_bytes()
}

fn half(bytes: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]))
}

/// One reference block decoder, writing block_len values.
type RefDecoder = fn(&[u8], &mut [f32]);

/// Decodes two blocks of every quant type with random bytes and compares
/// against the independent reference; also a non-block-aligned element count
/// to pin the final-block truncation.
#[test]
fn block_decoders_match_reference() {
    let quants: &[(GgmlType, RefDecoder)] = &[
        (GgmlType::Q8_0, ref_q8_0),
        (GgmlType::Q4_0, ref_q4_0),
        (GgmlType::Q4_1, ref_q4_1),
        (GgmlType::Q5_0, ref_q5_0),
        (GgmlType::Q5_1, ref_q5_1),
        (GgmlType::Q2K, ref_q2k),
        (GgmlType::Q3K, ref_q3k),
        (GgmlType::Q4K, ref_q4k),
        (GgmlType::Q5K, ref_q5k),
        (GgmlType::Q6K, ref_q6k),
    ];
    let mut rng = Bytes(7);
    for &(ty, reference) in quants {
        let bb = ty.block_bytes();
        let bl = ty.block_len();
        let raw = rng.take(bb * 2);
        let mut want = vec![0f32; bl * 2];
        reference(&raw[..bb], &mut want[..bl]);
        reference(&raw[bb..], &mut want[bl..]);

        let got = to_f32(ty, &raw, bl * 2).unwrap();
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let diff = (g - w).abs();
            assert!(
                diff <= w.abs() * 1e-4 + 1e-30,
                "{ty:?} element {i}: decoder {g} vs reference {w}"
            );
        }

        // Ask for fewer elements than the last block holds.
        let truncated = to_f32(ty, &raw, bl + 5).unwrap();
        assert_eq!(truncated.len(), bl + 5);
        assert_eq!(&truncated[..bl], &want[..bl]);
    }
}

#[test]
fn float_types_decode_directly() {
    let vals = [1.5f32, -2.25, 0.0, 100.0];
    let f32_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(to_f32(GgmlType::F32, &f32_bytes, 4).unwrap(), vals);

    let bf: Vec<u8> = vals
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    assert_eq!(
        to_f32(GgmlType::Bf16, &bf, 4).unwrap(),
        vals,
        "these values are bf16-exact"
    );

    let f16_bytes: Vec<u8> = [1.0f32, -2.0, 0.0, 2.5]
        .iter()
        .flat_map(|v| half_bits(*v))
        .collect();
    assert_eq!(
        to_f32(GgmlType::F16, &f16_bytes, 4).unwrap(),
        vec![1.0, -2.0, 0.0, 2.5]
    );
}

#[test]
fn truncated_input_fails_fast() {
    let err = to_f32(GgmlType::Q8_0, &[0u8; 10], 32).unwrap_err();
    assert!(err.to_string().contains("truncated"), "{err}");
}

#[test]
fn tensor_data_materializes_f32() {
    assert_eq!(TensorData::F32(vec![1.0, -2.0]).into_f32(), vec![1.0, -2.0]);
    let bf16_of_one = (1.0f32.to_bits() >> 16) as u16;
    let bf16_of_pi = (std::f32::consts::PI.to_bits() >> 16) as u16;
    let bytes: Vec<u8> = [bf16_of_one, bf16_of_pi]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let got = TensorData::Bf16(bytes).into_f32();
    assert_eq!(got[0], 1.0);
    assert!(
        (got[1] - 3.140625).abs() < 1e-6,
        "bf16 pi rounds to {}",
        got[1]
    );
}

// ---------------------------------------------------------------- references
// Each reference indexes the flat output directly from the value position,
// unlike the production decoders which stream per sub-block.

fn ref_q8_0(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for (i, out) in y.iter_mut().enumerate() {
        *out = (b[2 + i] as i8) as f32 * d;
    }
}

fn nibble(qs: &[u8], i: usize) -> u32 {
    if i < 16 {
        (qs[i] & 0xf) as u32
    } else {
        (qs[i - 16] >> 4) as u32
    }
}

fn ref_q4_0(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    for (i, out) in y.iter_mut().enumerate() {
        *out = (nibble(&b[2..18], i) as i32 - 8) as f32 * d;
    }
}

fn ref_q4_1(b: &[u8], y: &mut [f32]) {
    let (d, m) = (half(b, 0), half(b, 2));
    for (i, out) in y.iter_mut().enumerate() {
        *out = nibble(&b[4..20], i) as f32 * d + m;
    }
}

/// 5th bit per value comes from the qh bit plane: bit i for values 0..16,
/// bit i+16 for values 16..32.
fn fifth(qh: u32, i: usize) -> u32 {
    ((qh >> i) & 1) << 4
}

fn ref_q5_0(b: &[u8], y: &mut [f32]) {
    let d = half(b, 0);
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    for (i, out) in y.iter_mut().enumerate() {
        let q = nibble(&b[6..22], i) | fifth(qh, i);
        *out = (q as i32 - 16) as f32 * d;
    }
}

fn ref_q5_1(b: &[u8], y: &mut [f32]) {
    let (d, m) = (half(b, 0), half(b, 2));
    let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
    for (i, out) in y.iter_mut().enumerate() {
        let q = nibble(&b[8..24], i) | fifth(qh, i);
        *out = q as f32 * d + m;
    }
}

// K-quants: 256 values per block. For value v, (n, j, l) locate its 16-wide
// sub-block: n = v/128 half-block, j = 0..4 bit-shift stage, l = 0..32 pair
// position. Group index g selects the per-sub-block scale.

fn ref_q2k(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (half(b, 68), half(b, 70));
    let (scales, qs) = (&b[0..16], &b[16..80]);
    for (v, out) in y.iter_mut().enumerate() {
        let (n, r) = ((v / 128) * 32, v % 128); // byte half within qs
        let (j, l) = (r / 32, r % 32);
        let g = (n / 32) * 8 + 2 * j + l / 16;
        let q = (qs[n + l % 16 + 16 * (l / 16)] >> (2 * j)) & 3;
        *out = d * (scales[g] & 0xf) as f32 * q as f32 - dmin * (scales[g] >> 4) as f32;
    }
}

/// The 12 raw scale bytes hold 16 unsigned 6-bit scales: four bits from one
/// byte, two from another, per the ggml packing.
fn q3k_scale(raw: &[u8], j: usize) -> i32 {
    let s = match j {
        0..=3 => (raw[j] & 0xf) | ((raw[8 + j] & 0x3) << 4),
        4..=7 => (raw[j] & 0xf) | (((raw[j + 4] >> 2) & 0x3) << 4),
        8..=11 => ((raw[j - 8] >> 4) & 0xf) | (((raw[j] >> 4) & 0x3) << 4),
        _ => ((raw[j - 8] >> 4) & 0xf) | (((raw[j - 4] >> 6) & 0x3) << 4),
    };
    s as i32 - 32
}

fn ref_q3k(b: &[u8], y: &mut [f32]) {
    let d = half(b, 108);
    let (hmask, qs, raw) = (&b[0..32], &b[32..96], &b[96..108]);
    for (v, out) in y.iter_mut().enumerate() {
        let (n, r) = ((v / 128) * 32, v % 128);
        let (j, l) = (r / 32, r % 32);
        let qb = n + l % 16 + 16 * (l / 16); // qs advances with the half...
        let hb = l % 16 + 16 * (l / 16); // ...hmask is reused for both halves
        let g = (n / 32) * 8 + 2 * j + l / 16;
        let low = (qs[qb] >> (2 * j)) & 3;
        // High bits use planes 0..4 for the first half, 4..8 for the second.
        let high_kept = hmask[hb] & (1 << (j + 4 * (v / 128))) != 0;
        *out = d * q3k_scale(raw, g) as f32 * (low as i32 - if high_kept { 0 } else { 4 }) as f32;
    }
}

/// 6-bit scale / 6-bit min pair for sub-block j of q4k/q5k.
fn k4_scale_min(s: &[u8], j: usize) -> (u32, u32) {
    if j < 4 {
        ((s[j] & 63) as u32, (s[j + 4] & 63) as u32)
    } else {
        let d = (s[j + 4] & 0xf) | ((s[j - 4] >> 6) << 4);
        let m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
        (d as u32, m as u32)
    }
}

fn ref_q4k(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (half(b, 0), half(b, 2));
    let (scales, qs) = (&b[4..16], &b[16..144]);
    for (v, out) in y.iter_mut().enumerate() {
        let (j, r) = (v / 64, v % 64);
        let g = 2 * j + r / 32;
        let q = if r < 32 {
            qs[32 * j + r] & 0xf
        } else {
            qs[32 * j + r - 32] >> 4
        };
        let (sc, m) = k4_scale_min(scales, g);
        *out = d * sc as f32 * q as f32 - dmin * m as f32;
    }
}

fn ref_q5k(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (half(b, 0), half(b, 2));
    let (scales, qh, ql) = (&b[4..16], &b[16..48], &b[48..176]);
    for (v, out) in y.iter_mut().enumerate() {
        let (j, r) = (v / 64, v % 64);
        let g = 2 * j + r / 32;
        let byte = 32 * j + r % 32;
        let (low, hi_bit) = if r < 32 {
            (ql[byte] & 0xf, 1u8 << (2 * j))
        } else {
            (ql[byte] >> 4, 2u8 << (2 * j))
        };
        let q = low + if qh[v % 32] & hi_bit != 0 { 16 } else { 0 };
        let (sc, m) = k4_scale_min(scales, g);
        *out = d * sc as f32 * q as f32 - dmin * m as f32;
    }
}

fn ref_q6k(b: &[u8], y: &mut [f32]) {
    let d = half(b, 208);
    let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
    for (v, out) in y.iter_mut().enumerate() {
        let (n, r) = (v / 128, v % 128);
        let (q, l) = (r / 32, r % 32);
        // Quarters 0/1 read the low nibble, 2/3 the high; odd quarters take
        // the qs byte 32 ahead. Scale and qh shift advance by two per quarter.
        let qlb = ql[64 * n + l + 32 * (q % 2)];
        let low = if q >= 2 { qlb >> 4 } else { qlb & 0xf };
        let qval = (low | (((qh[32 * n + l] >> (2 * q)) & 3) << 4)) as i32 - 32;
        *out = d * (sc[8 * n + l / 16 + 2 * q] as i8) as f32 * qval as f32;
    }
}

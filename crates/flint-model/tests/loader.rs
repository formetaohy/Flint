use flint_model::quant::{choose_group, quantize};

#[test]
fn group_prefers_128_then_falls_back() {
    assert_eq!(choose_group(128).unwrap(), 128);
    assert_eq!(choose_group(256).unwrap(), 128);
    assert_eq!(
        choose_group(960).unwrap(),
        64,
        "SmolLM-style width is not a multiple of 128"
    );
    assert_eq!(choose_group(160).unwrap(), 32);
    assert_eq!(choose_group(32).unwrap(), 32);
}

#[test]
fn group_rejects_unquantizable_width() {
    let err = choose_group(16).unwrap_err();
    assert!(err.to_string().contains("not a multiple of 32"));
}

#[test]
fn quantize_matches_hand_computed() {
    #[rustfmt::skip]
    let data: [f32; 64] = {
        let mut d = [0.0f32; 64];
        d[0] = 1.0; d[1] = -0.25;
        d[32] = -2.0; d[33] = 0.25; d[34] = 0.5; d[35] = -0.125;
        d
    };
    let (bytes, scales) = quantize(&data, 2, 32, 32);

    assert_eq!(scales, vec![1.0f32 / 127.0, 2.0f32 / 127.0]);
    #[rustfmt::skip]
    let expect: [i8; 64] = {
        let mut e = [0i8; 64];

        e[0] = 127; e[1] = -32;
        e[16] = -127; e[17] = 16; e[18] = 32; e[19] = -8;
        e
    };
    let expect_bytes: Vec<u8> = expect.iter().map(|q| *q as u8).collect();
    assert_eq!(bytes, expect_bytes);
}

#[test]
fn quantize_roundtrip_stays_within_half_a_step() {
    let mut seed = 11u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f32 / (1u32 << 31) as f32 * 2.0 - 1.0
    };
    let (rows, cols, group) = (3usize, 256usize, 128usize);
    let data: Vec<f32> = (0..rows * cols).map(|_| next()).collect();
    let (bytes, scales) = quantize(&data, rows, cols, group);

    assert_eq!(bytes.len(), rows * cols);
    assert_eq!(scales.len(), rows * cols / group);
    for (i, &b) in bytes.iter().enumerate() {
        let kb = i / (rows * 16);
        let rem = i % (rows * 16);
        let (r, ii) = (rem / 16, rem % 16);
        let col = kb * 16 + ii;
        let scale = scales[(col / group) * rows + r];
        let deq = (b as i8) as f32 * scale;
        let err = (deq - data[r * cols + col]).abs();
        assert!(
            err <= scale * 0.5 + 1e-6,
            "index {i}: dequant {deq} vs {}",
            data[r * cols + col]
        );
    }
}

#[test]
#[should_panic(expected = "multiple of the group size")]
fn quantize_rejects_misaligned_width() {
    quantize(&[0.0; 48], 1, 48, 32);
}

#[test]
#[should_panic(expected = "multiple of 16 (vec4 blocks)")]
fn quantize_rejects_non_vec4_width() {
    quantize(&[0.0; 4], 1, 4, 2);
}

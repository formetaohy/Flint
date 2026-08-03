//! Backend resource layer: buffer factories, readback, copies, tensor
//! geometry, weight invariants and fail-fast error paths.

use flint_backend::{Backend, Binding, Pass};
use flint_tensor::{DType, Weight};

#[test]
fn f32_roundtrip() {
    let backend = Backend::new().unwrap();
    let data = [1.5f32, -2.25, 0.0, 1e30];
    let t = backend.tensor_f32(&data, vec![4], "t");
    assert_eq!(backend.read_f32(&t.buf, 0, 4).unwrap(), data);
}

#[test]
fn read_f32_honors_byte_offset() {
    let backend = Backend::new().unwrap();
    let t = backend.tensor_f32(&[1.0, 2.0, 3.0, 4.0], vec![4], "t");
    assert_eq!(backend.read_f32(&t.buf, 8, 2).unwrap(), vec![3.0, 4.0]);
}

#[test]
fn zero_fill_resets_a_written_tensor() {
    let backend = Backend::new().unwrap();
    let t = backend.tensor_f32(&[9.0; 4], vec![4], "t");
    backend.zero_fill(&t);
    assert_eq!(backend.read_f32(&t.buf, 0, 4).unwrap(), vec![0.0; 4]);
}

#[test]
fn copy_duplicates_contents() {
    let backend = Backend::new().unwrap();
    let src = backend.tensor_f32(&[5.0, 6.0], vec![2], "src");
    let dst = backend.zero_tensor(&[2], "dst");
    backend.copy(&src, &dst);
    assert_eq!(backend.read_f32(&dst.buf, 0, 2).unwrap(), vec![5.0, 6.0]);
}

#[test]
fn write_u32_lands_little_endian() {
    let backend = Backend::new().unwrap();
    let buf = backend.storage(8, "ids");
    backend.write_u32(&buf, &[f32::to_bits(1.0), 42]);
    let got = backend.read_f32(&buf, 0, 2).unwrap();
    assert_eq!(got[0], 1.0);
    assert_eq!(got[1], f32::from_bits(42));
}

#[test]
fn tensor_geometry_per_dtype() {
    let backend = Backend::new().unwrap();
    let f = backend.tensor_f32(&[0.0; 6], vec![2, 3], "f");
    assert_eq!(f.numel(), 6);
    assert_eq!(f.byte_len(), 24);

    let b = backend.tensor_bf16(&[0u8; 12], vec![2, 3], "b").unwrap();
    assert_eq!(b.dtype, DType::Bf16Packed);
    assert_eq!(b.byte_len(), 12);

    let i = backend.tensor_i8(&[0u8; 8], vec![2, 4], "i");
    assert_eq!(i.byte_len(), 8);
}

#[test]
fn bf16_odd_byte_count_fails_fast() {
    let backend = Backend::new().unwrap();
    let err = backend
        .tensor_bf16(&[0u8; 3], vec![1], "bad")
        .err()
        .expect("odd byte count must fail");
    assert!(err.to_string().contains("odd bf16"), "{err}");
}

#[test]
#[should_panic(expected = "multiple of 4")]
fn i8_count_must_align_to_u32_words() {
    let backend = Backend::new().unwrap();
    backend.tensor_i8(&[0u8; 3], vec![3], "bad");
}

#[test]
fn unknown_shader_is_an_error_not_a_panic() {
    let mut backend = Backend::new().unwrap();
    let t = backend.zero_tensor(&[1], "t");
    let mut enc = backend.encoder();
    let mut pass = Pass::begin(&mut enc, "k");
    let err = backend
        .dispatch(&mut pass, "nope", &[], &[Binding::Full(&t)], [1, 1, 1])
        .unwrap_err();
    assert!(err.to_string().contains("unknown shader"), "{err}");
}

#[test]
fn weight_invariants() {
    let backend = Backend::new().unwrap();
    let f = backend.tensor_f32(&[0.0; 4], vec![2, 2], "f");
    let b = backend.tensor_bf16(&[0u8; 8], vec![2, 2], "b").unwrap();
    let i = backend.tensor_i8(&[0u8; 4], vec![2, 2], "i");
    let s = backend.tensor_f32(&[1.0; 2], vec![2, 1], "s");

    assert!(matches!(Weight::plain(f), Weight::Plain(_)));
    assert!(matches!(Weight::plain(b), Weight::Plain(_)));
    match Weight::quant(i, s, 128) {
        Weight::Quantized { group, .. } => assert_eq!(group, 128),
        Weight::Plain(_) => unreachable!(),
    }

    // Accessors across variants: plain weights carry no scale and default
    // the group to the preferred 128; quantized weights report their own.
    let f = backend.tensor_f32(&[0.0; 4], vec![2, 2], "f");
    let w = Weight::plain(f);
    assert!(w.scale().is_none());
    assert_eq!(w.group(), 128);
    assert_eq!(w.tensor().shape, vec![2, 2]);
    let i = backend.tensor_i8(&[0u8; 4], vec![2, 2], "i");
    let s = backend.tensor_f32(&[1.0; 1], vec![1, 1], "s");
    let q = Weight::quant(i, s, 32);
    assert_eq!(q.group(), 32);
    assert!(q.scale().is_some());
}

#[test]
#[should_panic(expected = "must be f32 or bf16")]
fn plain_weight_rejects_i8() {
    let backend = Backend::new().unwrap();
    let i = backend.tensor_i8(&[0u8; 4], vec![2, 2], "i");
    Weight::plain(i);
}

#[test]
#[should_panic(expected = "must be i8")]
fn quant_weight_rejects_floats() {
    let backend = Backend::new().unwrap();
    let f = backend.tensor_f32(&[0.0; 4], vec![2, 2], "f");
    let s = backend.tensor_f32(&[1.0], vec![1], "s");
    Weight::quant(f, s, 128);
}

#[test]
fn adapter_name_is_reported() {
    let backend = Backend::new().unwrap();
    assert!(!backend.adapter_name().is_empty());
}

#[test]
fn dummy_scale_is_a_single_element_tensor() {
    let backend = Backend::new().unwrap();
    let d = backend.dummy_scale();
    assert_eq!(d.numel(), 1);
    assert_eq!(d.dtype, DType::F32);
}

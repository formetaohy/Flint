//! Safetensors checkpoint source: sharded index dispatch, single-shard
//! fallback, dtype decoding (f32 / bf16 / f16), config.json and the
//! fail-fast error paths. Round-trips through `write_tensors` so no external
//! files are needed.

use std::path::PathBuf;

use safetensors::serialize;
use safetensors::tensor::{Dtype, TensorView};

use flint_checkpoint::{Checkpoint, Safetensors, write_tensors};

fn tmp_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("flint-safetensors-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn err_str<T>(r: flint_error::Result<T>) -> String {
    r.err().expect("expected an error").to_string()
}

/// One tensor as (name, shape, little-endian bytes, bf16?).
fn tensor(name: &str, shape: &[u32], data: &[f32]) -> (String, Vec<u32>, Vec<u8>, bool) {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    (name.to_string(), shape.to_vec(), bytes, false)
}

#[test]
fn single_shard_roundtrip_without_index() {
    let dir = tmp_dir("single");
    let tensors = vec![
        tensor("a", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        tensor("b", &[4], &[-1.0, 0.0, 2.5, 1e30]),
    ];
    write_tensors(&dir.join("model.safetensors"), &tensors).unwrap();
    std::fs::write(dir.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();

    let st = Safetensors::open(&dir).unwrap();
    assert_eq!(st.kind(), "safetensors");

    let mut names = st.names();
    names.sort();
    assert_eq!(names, vec!["a".to_string(), "b".to_string()]);

    let a = st.read("a").unwrap();
    assert_eq!(a.shape, vec![2, 3]);
    assert_eq!(a.data.into_f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let b = st.read("b").unwrap();
    assert_eq!(b.data.into_f32(), vec![-1.0, 0.0, 2.5, 1e30]);

    let cfg = st.config_json().unwrap().unwrap();
    assert_eq!(cfg["model_type"], "llama");

    assert!(
        err_str(st.read("nope")).contains("not in checkpoint index"),
        "unknown tensor name fails fast"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn bf16_and_f16_decode_to_f32() {
    let dir = tmp_dir("f16");
    // bf16 via write_tensors' bf16 flag.
    let bf16_bytes: Vec<u8> = [1.0f32, -2.0, 0.5]
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    write_tensors(
        &dir.join("model.safetensors"),
        &[("b".to_string(), vec![3], bf16_bytes, true)],
    )
    .unwrap();

    // f16 via the raw serialize path (write_tensors never emits f16).
    let f16_bytes: Vec<u8> = [1.0f32, -2.0, 2.5]
        .iter()
        .flat_map(|v| f16_bits(*v).to_le_bytes())
        .collect();
    let view = TensorView::new(Dtype::F16, vec![3], &f16_bytes).unwrap();
    // Both tensors in one shard: bf16 (write_tensors) plus f16 (raw serialize).
    let st_bytes = std::fs::read(dir.join("model.safetensors")).unwrap();
    let both = serialize(
        vec![
            ("b".to_string(), tensor_view_from(&st_bytes, "b")),
            ("f".to_string(), view),
        ],
        None,
    )
    .unwrap();
    std::fs::write(dir.join("model.safetensors"), both).unwrap();

    let st = Safetensors::open(&dir).unwrap();
    let b = st.read("b").unwrap();
    assert_eq!(b.data.into_f32(), vec![1.0, -2.0, 0.5]);

    let f = st.read("f").unwrap();
    assert_eq!(f.data.into_f32(), vec![1.0, -2.0, 2.5]);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn sharded_index_dispatches_across_files() {
    let dir = tmp_dir("sharded");
    write_tensors(
        &dir.join("model-00001-of-00002.safetensors"),
        &[tensor("layers.0.q.weight", &[16, 64], &[1.0; 1024])],
    )
    .unwrap();
    write_tensors(
        &dir.join("model-00002-of-00002.safetensors"),
        &[tensor("norm.weight", &[64], &[2.0; 64])],
    )
    .unwrap();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{
            "weight_map": {
                "layers.0.q.weight": "model-00001-of-00002.safetensors",
                "norm.weight": "model-00002-of-00002.safetensors"
            }
        }"#,
    )
    .unwrap();

    let st = Safetensors::open(&dir).unwrap();
    assert_eq!(st.names().len(), 2);
    let q = st.read("layers.0.q.weight").unwrap();
    assert_eq!(q.shape, vec![16, 64]);
    assert_eq!(q.data.into_f32()[0], 1.0);
    let n = st.read("norm.weight").unwrap();
    assert_eq!(n.data.into_f32()[63], 2.0);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn fail_fast_paths() {
    let dir = tmp_dir("fail");

    // No index and no single shard.
    assert!(
        err_str(Safetensors::open(&dir)).contains("no safetensors index"),
        "empty dir must fail"
    );

    // Corrupt index JSON.
    std::fs::write(dir.join("model.safetensors.index.json"), "not json").unwrap();
    assert!(Safetensors::open(&dir).is_err());

    // Index referencing a missing shard.
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{"weight_map": {"a": "gone.safetensors"}}"#,
    )
    .unwrap();
    let st = Safetensors::open(&dir).unwrap();
    assert!(
        err_str(st.read("a")).contains("missing shard"),
        "missing shard"
    );

    // Unsupported dtype (F64 is never produced by Flint writers).
    let view = TensorView::new(Dtype::F64, vec![2], &[0u8; 16]).unwrap();
    let bytes = serialize(vec![("d".to_string(), view)], None).unwrap();
    std::fs::write(dir.join("d.safetensors"), bytes).unwrap();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{"weight_map": {"d": "d.safetensors"}}"#,
    )
    .unwrap();
    let st = Safetensors::open(&dir).unwrap();
    assert!(
        err_str(st.read("d")).contains("unsupported safetensors dtype"),
        "F64 must be rejected"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn open_requires_something_parseable() {
    let dir = tmp_dir("empty-file");
    std::fs::write(dir.join("model.safetensors"), b"garbage").unwrap();
    assert!(Safetensors::open(&dir).is_err(), "garbage shard must fail");
    std::fs::remove_dir_all(&dir).ok();
}

/// Re-extracts one tensor view from an existing serialized file so the bf16
/// tensor written by `write_tensors` can be merged with the f16 one.
fn tensor_view_from<'a>(bytes: &'a [u8], name: &str) -> safetensors::tensor::TensorView<'a> {
    let st = safetensors::SafeTensors::deserialize(bytes).unwrap();
    st.tensor(name).unwrap()
}

/// Nearest-even f32 -> f16, enough for the exact small values used here.
fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (b >> 13) & 0x3ff;
    if exp <= 0 {
        0
    } else if exp >= 0x1f {
        0x7c00
    } else {
        sign | ((exp as u16) << 10) | mant as u16
    }
}

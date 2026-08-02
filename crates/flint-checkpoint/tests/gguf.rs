//! GGUF container reading against synthesized files: metadata value types,
//! custom alignment, tensor info + dequantization, dim reversal and the
//! format-detection dispatch.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use flint_checkpoint::{Checkpoint, Gguf, MetaVal};

// ---------------------------------------------------------------- writer

fn w_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn w_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn w_str(out: &mut Vec<u8>, s: &str) {
    w_u64(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}
fn w_kv_str(out: &mut Vec<u8>, key: &str, val: &str) {
    w_str(out, key);
    w_u32(out, 8);
    w_str(out, val);
}
fn w_kv_u32(out: &mut Vec<u8>, key: &str, val: u32) {
    w_str(out, key);
    w_u32(out, 4);
    w_u32(out, val);
}

fn f16_bytes(v: f32) -> [u8; 2] {
    let b = v.to_bits();
    let h = ((b >> 16) & 0x8000) as u16
        | ((((b >> 23) & 0xff) as u16).wrapping_sub(127 - 15) << 10)
        | ((b >> 13) & 0x3ff) as u16;
    h.to_le_bytes()
}

/// Builds a GGUF v3 file: mixed metadata plus an f32 [3,2] matrix and a
/// q8_0 [32] vector, under a non-default alignment of 64.
fn synth_gguf() -> Vec<u8> {
    let mut h = Vec::new();
    w_u32(&mut h, 0x4655_4747); // "GGUF"
    w_u32(&mut h, 3);
    w_u64(&mut h, 2); // tensors
    w_u64(&mut h, 8); // kv pairs

    w_kv_str(&mut h, "general.architecture", "llama");
    w_kv_u32(&mut h, "general.alignment", 64);
    w_kv_u32(&mut h, "llama.block_count", 24);
    w_str(&mut h, "test.f32");
    w_u32(&mut h, 6);
    h.extend_from_slice(&1.5f32.to_le_bytes());
    w_str(&mut h, "test.bool");
    w_u32(&mut h, 7);
    h.push(1);
    w_str(&mut h, "test.arr_str");
    w_u32(&mut h, 9);
    w_u32(&mut h, 8);
    w_u64(&mut h, 2);
    w_str(&mut h, "a");
    w_str(&mut h, "b");
    w_str(&mut h, "test.arr_u32");
    w_u32(&mut h, 9);
    w_u32(&mut h, 4);
    w_u64(&mut h, 3);
    for v in [10u32, 20, 30] {
        w_u32(&mut h, v);
    }
    w_str(&mut h, "test.i8");
    w_u32(&mut h, 1);
    h.push((-5i8) as u8);

    // Tensor infos: dims are fastest-first, offsets relative to data start.
    w_str(&mut h, "t_f32");
    w_u32(&mut h, 2);
    w_u64(&mut h, 3); // fastest dim -> reported shape [2, 3]
    w_u64(&mut h, 2);
    w_u32(&mut h, 0); // F32
    w_u64(&mut h, 0);

    w_str(&mut h, "t_q8");
    w_u32(&mut h, 1);
    w_u64(&mut h, 32);
    w_u32(&mut h, 8); // Q8_0
    w_u64(&mut h, 64); // after the 24-byte f32 matrix, aligned to 64

    while h.len() % 64 != 0 {
        h.push(0);
    }
    for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        h.extend_from_slice(&v.to_le_bytes());
    }
    while h.len() % 64 != 0 {
        h.push(0);
    }
    h.extend_from_slice(&f16_bytes(2.0)); // d = 2.0
    h.extend((0i8..32).map(|q| q as u8)); // qs = 0..32 -> values 0,2,..,62

    h
}

fn tmp_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("flint-gguf-{}-{}", test, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_synth(test: &str, file: &str) -> PathBuf {
    let dir = tmp_dir(test);
    let path = dir.join(file);
    std::fs::write(&path, synth_gguf()).unwrap();
    path
}

/// Extracts the error text (the Ok types here do not implement Debug).
fn err_str<T>(r: flint_error::Result<T>) -> String {
    r.err().expect("expected an error").to_string()
}

// ---------------------------------------------------------------- tests

#[test]
fn reads_metadata_and_tensors() {
    let path = write_synth("main", "m.gguf");
    let gguf = Gguf::open(&path).unwrap();

    assert_eq!(gguf.kind(), "gguf");
    assert!(gguf.config_json().unwrap().is_none());

    let meta = gguf.metadata();
    assert_eq!(meta.str("general.architecture"), Some("llama"));
    assert_eq!(meta.u32("llama.block_count"), Some(24));
    assert_eq!(
        meta.f64("llama.block_count"),
        Some(24.0),
        "ints coerce to f64"
    );
    assert_eq!(meta.get("test.bool"), Some(&MetaVal::Bool(true)));
    assert_eq!(meta.f64("test.f32"), Some(1.5));
    assert_eq!(meta.str_array("test.arr_str").unwrap(), vec!["a", "b"]);
    assert_eq!(meta.u32_array("test.arr_u32").unwrap(), vec![10, 20, 30]);
    assert_eq!(meta.u64("test.i8"), None, "negative int is not a u64");
    assert_eq!(meta.str("missing"), None);

    let mut names = gguf.names();
    names.sort();
    assert_eq!(names, vec!["t_f32".to_string(), "t_q8".to_string()]);

    let f = gguf.read("t_f32").unwrap();
    assert_eq!(f.shape, vec![2, 3], "dims reverse to [N, K]");
    assert_eq!(f.data.into_f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let q = gguf.read("t_q8").unwrap();
    assert_eq!(q.shape, vec![32]);
    let vals = q.data.into_f32();
    let want: Vec<f32> = (0..32).map(|i| i as f32 * 2.0).collect();
    assert_eq!(vals, want);

    assert!(gguf.read("nope").is_err());
    std::fs::remove_dir_all(path.parent().unwrap()).ok();
}

#[test]
fn rejects_bad_magic_and_truncation() {
    let dir = tmp_dir("bad");
    let bad = dir.join("bad.gguf");
    std::fs::write(&bad, b"not a gguf file at all").unwrap();
    assert!(err_str(Gguf::open(&bad)).contains("not a GGUF"));

    let mut bytes = synth_gguf();
    bytes.truncate(40);
    let short = dir.join("short.gguf");
    std::fs::write(&short, bytes).unwrap();
    assert!(Gguf::open(&short).is_err(), "truncated header must fail");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn open_dispatches_on_directory_contents() {
    // A lone .gguf wins.
    let dir = tmp_dir("open");
    std::fs::write(dir.join("m.gguf"), synth_gguf()).unwrap();
    assert_eq!(flint_checkpoint::open(&dir).unwrap().kind(), "gguf");

    // Split shards fail fast.
    std::fs::write(dir.join("m-00001.gguf"), synth_gguf()).unwrap();
    assert!(
        err_str(flint_checkpoint::open(&dir)).contains("shards"),
        "split shards fail fast"
    );
    std::fs::remove_dir_all(&dir).ok();

    // Neither format present.
    let empty = tmp_dir("empty");
    assert!(flint_checkpoint::open(&empty).is_err());
    std::fs::remove_dir_all(&empty).ok();

    assert!(flint_checkpoint::open(Path::new("./no-such-dir-flint")).is_err());
}

#[test]
fn metadata_constructs_from_kv() {
    let meta = flint_checkpoint::Metadata::new(HashMap::from([
        ("s".to_string(), MetaVal::Str("v".into())),
        ("n".to_string(), MetaVal::UInt(9)),
    ]));
    assert_eq!(meta.str("s"), Some("v"));
    assert_eq!(meta.u32("n"), Some(9));
    assert!(!meta.is_empty());
    assert!(flint_checkpoint::Metadata::default().is_empty());
}

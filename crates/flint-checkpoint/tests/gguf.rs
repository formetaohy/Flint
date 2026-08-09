use std::path::{Path, PathBuf};

use flint_checkpoint::{Checkpoint, CheckpointKind, Gguf, MetaVal};

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

fn synth_gguf() -> Vec<u8> {
    let mut h = Vec::new();
    w_u32(&mut h, 0x4655_4747); 
    w_u32(&mut h, 3);
    w_u64(&mut h, 2); 
    w_u64(&mut h, 8); 

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

    w_str(&mut h, "t_f32");
    w_u32(&mut h, 2);
    w_u64(&mut h, 3); 
    w_u64(&mut h, 2);
    w_u32(&mut h, 0); 
    w_u64(&mut h, 0);

    w_str(&mut h, "t_q8");
    w_u32(&mut h, 1);
    w_u64(&mut h, 32);
    w_u32(&mut h, 8); 
    w_u64(&mut h, 64); 

    while h.len() % 64 != 0 {
        h.push(0);
    }
    for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        h.extend_from_slice(&v.to_le_bytes());
    }
    while h.len() % 64 != 0 {
        h.push(0);
    }
    h.extend_from_slice(&f16_bytes(2.0)); 
    h.extend((0i8..32).map(|q| q as u8)); 

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

fn err_str<T>(r: flint_error::Result<T>) -> String {
    r.err().expect("expected an error").to_string()
}

#[test]
fn reads_metadata_and_tensors() {
    let path = write_synth("main", "m.gguf");
    let gguf = Gguf::open(&path).unwrap();

    assert_eq!(gguf.kind(), CheckpointKind::Gguf);
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

    let dir = tmp_dir("open");
    std::fs::write(dir.join("m.gguf"), synth_gguf()).unwrap();
    assert_eq!(flint_checkpoint::open(&dir).unwrap().kind(), CheckpointKind::Gguf);

    std::fs::write(dir.join("m-00001.gguf"), synth_gguf()).unwrap();
    assert!(
        err_str(flint_checkpoint::open(&dir)).contains("shards"),
        "split shards fail fast"
    );
    std::fs::remove_dir_all(&dir).ok();

    let empty = tmp_dir("empty");
    assert!(flint_checkpoint::open(&empty).is_err());
    std::fs::remove_dir_all(&empty).ok();

    assert!(flint_checkpoint::open(Path::new("./no-such-dir-flint")).is_err());
}

fn synth_full_types() -> Vec<u8> {
    let mut h = Vec::new();
    w_u32(&mut h, 0x4655_4747);
    w_u32(&mut h, 3);
    w_u64(&mut h, 0); 
    w_u64(&mut h, 9); 

    let kv = |out: &mut Vec<u8>, key: &str, ty: u32, body: &[u8]| {
        w_str(out, key);
        w_u32(out, ty);
        out.extend_from_slice(body);
    };
    kv(&mut h, "t_u8", 0, &[200]);
    kv(&mut h, "t_i8", 1, &[(-7i8) as u8]);
    kv(&mut h, "t_u16", 2, &500u16.to_le_bytes());
    kv(&mut h, "t_i16", 3, &(-500i16).to_le_bytes());
    kv(&mut h, "t_i32", 5, &(-1_000_000i32).to_le_bytes());
    kv(&mut h, "t_u64", 10, &(1u64 << 40).to_le_bytes());
    kv(&mut h, "t_i64", 11, &(-(1i64 << 40)).to_le_bytes());
    kv(&mut h, "t_f64", 12, &std::f64::consts::PI.to_le_bytes());

    let mut arr = Vec::new();
    w_u32(&mut arr, 6);
    w_u64(&mut arr, 2);
    arr.extend_from_slice(&1.5f32.to_le_bytes());
    arr.extend_from_slice(&(-2.0f32).to_le_bytes());
    kv(&mut h, "t_arr_f32", 9, &arr);
    h
}

#[test]
fn reads_every_metadata_value_type() {
    let dir = tmp_dir("types");
    let path = dir.join("m.gguf");
    std::fs::write(&path, synth_full_types()).unwrap();
    let gguf = Gguf::open(&path).unwrap();
    let meta = gguf.metadata();

    assert_eq!(meta.u64("t_u8"), Some(200));
    assert_eq!(meta.u64("t_i8"), None, "negative i8 is not a u64");
    assert_eq!(meta.u64("t_u16"), Some(500));
    assert_eq!(meta.u64("t_i16"), None, "negative i16 is not a u64");
    assert_eq!(meta.u64("t_u64"), Some(1u64 << 40));
    assert_eq!(meta.u64("t_i64"), None);
    assert_eq!(meta.u64("t_i32"), None);
    assert_eq!(meta.f64("t_f64"), Some(std::f64::consts::PI));
    assert_eq!(meta.f64("t_i32"), Some(-1_000_000.0), "ints coerce to f64");
    assert_eq!(meta.f64("t_u8"), Some(200.0));
    assert_eq!(meta.f64_array("t_arr_f32").unwrap(), vec![1.5, -2.0]);

    assert_eq!(meta.str("t_u8"), None);
    assert_eq!(meta.u32("t_f64"), None);
    assert_eq!(
        meta.u32_array("t_arr_f32"),
        None,
        "float array is not a u32 array"
    );
    assert_eq!(meta.str_array("t_u64"), None);
    assert_eq!(meta.f64_array("t_i8"), None);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn rejects_unknown_value_types_and_versions() {
    let dir = tmp_dir("bad-ty");

    let mut bad = synth_full_types();
    let tag_off = 8 + 8 + 8 + 8 + 4; 
    bad[tag_off] = 13;
    let p = dir.join("bad.gguf");
    std::fs::write(&p, &bad).unwrap();
    assert!(
        err_str(Gguf::open(&p)).contains("unknown GGUF metadata value type"),
        "unknown tag must fail"
    );

    let mut v4 = synth_gguf();
    v4[4..8].copy_from_slice(&4u32.to_le_bytes());
    let p = dir.join("v4.gguf");
    std::fs::write(&p, &v4).unwrap();
    assert!(err_str(Gguf::open(&p)).contains("unsupported GGUF version"));

    let mut v2 = synth_gguf();
    v2[4..8].copy_from_slice(&2u32.to_le_bytes());
    let p = dir.join("v2.gguf");
    std::fs::write(&p, &v2).unwrap();
    assert_eq!(
        Gguf::open(&p).unwrap().metadata().u32("llama.block_count"),
        Some(24)
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn writer_roundtrips_through_reader() {
    let dir = tmp_dir("writer");
    let mut w = flint_checkpoint::GgufWriter::new(32);
    w.kv_str("general.architecture", "llama");
    w.kv_u32("llama.block_count", 2);
    w.kv_f32("llama.rope.freq_base", 10000.0);
    w.kv_bool("llama.bool", true);
    w.kv_str_array("tokenizer.ggml.tokens", &["a", "b", "<|endoftext|>"]);
    w.kv_u32_array("tokenizer.ggml.token_type", &[0, 0, 3]);
    w.kv_f64_array("tokenizer.ggml.scores", &[1.5, 0.5, -1.0]);

    let f32_data: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
    let bf16_data: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
    let q8_data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 3.0).collect();
    w.tensor_f32("t_f32", &[2, 16], &f32_data);
    w.tensor_bf16("t_bf16", &[2, 32], &bf16_data);
    w.tensor_q8_0("t_q8", &[2, 32], &q8_data);
    std::fs::write(dir.join("m.gguf"), w.finish()).unwrap();

    let g = Gguf::open(&dir.join("m.gguf")).unwrap();
    assert_eq!(g.kind(), CheckpointKind::Gguf);
    assert_eq!(g.metadata().str("general.architecture"), Some("llama"));
    assert_eq!(g.metadata().u32("llama.block_count"), Some(2));
    assert_eq!(g.metadata().f64("llama.rope.freq_base"), Some(10000.0));
    assert_eq!(g.metadata().get("llama.bool"), Some(&MetaVal::Bool(true)));
    assert_eq!(
        g.metadata().str_array("tokenizer.ggml.tokens"),
        Some(vec!["a", "b", "<|endoftext|>"])
    );
    assert_eq!(
        g.metadata().u32_array("tokenizer.ggml.token_type"),
        Some(vec![0, 0, 3])
    );
    assert_eq!(
        g.metadata().f64_array("tokenizer.ggml.scores"),
        Some(vec![1.5, 0.5, -1.0])
    );

    let names = g.names();
    assert_eq!(names.len(), 3);

    let f32 = g.read("t_f32").unwrap();
    assert_eq!(f32.shape, vec![2, 16]);
    assert_eq!(f32.data.into_f32(), f32_data);

    let bf16 = g.read("t_bf16").unwrap();
    assert_eq!(bf16.shape, vec![2, 32]);
    let got: Vec<f32> = bf16
        .data
        .into_f32()
        .iter()
        .zip(&bf16_data)
        .map(|(a, b)| (a - b).abs())
        .collect();
    assert!(
        got.iter().all(|d| *d <= 2f32.powi(-9) * 32.0),
        "bf16 drift {:?}",
        got
    );

    let q8 = g.read("t_q8").unwrap();
    assert_eq!(q8.shape, vec![2, 32]);
    let got = q8.data.into_f32();
    for (a, b) in got.iter().zip(&q8_data) {

        assert!((a - b).abs() < 0.5, "{a} vs {b}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

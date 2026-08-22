use std::path::{Path, PathBuf};

use thuban_checkpoint::{Checkpoint, Gguf, MetaVal};
use thuban_tensor::Quant;

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
    let dir = std::env::temp_dir().join(format!("thuban-gguf-{}-{}", test, std::process::id()));
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

fn err_str<T>(r: thuban_error::Result<T>) -> String {
    r.err().expect("expected an error").to_string()
}

#[test]
fn reads_metadata_and_tensors() {
    let path = write_synth("main", "m.gguf");
    let gguf = Gguf::open(&path).unwrap();

    let meta = gguf.metadata().unwrap();
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
    assert_eq!(
        f.data.into_f32().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    let q = gguf.read("t_q8").unwrap();
    assert_eq!(q.shape, vec![32]);
    let vals = q.data.into_f32().unwrap();
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
    let gguf = thuban_checkpoint::open_checkpoint(&dir).unwrap();
    assert_eq!(
        gguf.metadata().unwrap().str("general.architecture"),
        Some("llama")
    );

    std::fs::write(dir.join("m-00001.gguf"), synth_gguf()).unwrap();
    assert!(
        err_str(thuban_checkpoint::open_checkpoint(&dir)).contains("shards"),
        "split shards fail fast"
    );
    std::fs::remove_dir_all(&dir).ok();

    let empty = tmp_dir("empty");
    assert!(
        err_str(thuban_checkpoint::open_checkpoint(&empty)).contains("no .gguf"),
        "a directory without GGUF files must fail"
    );
    std::fs::remove_dir_all(&empty).ok();

    assert!(thuban_checkpoint::open_checkpoint(Path::new("./no-such-dir-thuban")).is_err());
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
    let meta = gguf.metadata().unwrap();

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
        Gguf::open(&p)
            .unwrap()
            .metadata()
            .unwrap()
            .u32("llama.block_count"),
        Some(24)
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn writer_roundtrips_through_reader() {
    let dir = tmp_dir("writer");
    let mut w = thuban_checkpoint::GgufWriter::new(32);
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
    let mut q8_blocks = Vec::new();
    for blk in q8_data.chunks_exact(32) {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
        q8_blocks.extend_from_slice(&thuban_num::f32_to_f16(d).to_le_bytes());
        for v in blk {
            q8_blocks.push((v / d).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
    w.tensor_f32("t_f32", &[2, 16], &f32_data);
    w.tensor_bf16("t_bf16", &[2, 32], &bf16_data);
    w.tensor_raw("t_q8", &[2, 32], Quant::Q8_0, &q8_blocks);
    std::fs::write(dir.join("m.gguf"), w.finish()).unwrap();

    let g = Gguf::open(&dir.join("m.gguf")).unwrap();
    assert_eq!(
        g.metadata().unwrap().str("general.architecture"),
        Some("llama")
    );
    assert_eq!(g.metadata().unwrap().u32("llama.block_count"), Some(2));
    assert_eq!(
        g.metadata().unwrap().f64("llama.rope.freq_base"),
        Some(10000.0)
    );
    assert_eq!(
        g.metadata().unwrap().get("llama.bool"),
        Some(&MetaVal::Bool(true))
    );
    assert_eq!(
        g.metadata().unwrap().str_array("tokenizer.ggml.tokens"),
        Some(vec!["a", "b", "<|endoftext|>"])
    );
    assert_eq!(
        g.metadata().unwrap().u32_array("tokenizer.ggml.token_type"),
        Some(vec![0, 0, 3])
    );
    assert_eq!(
        g.metadata().unwrap().f64_array("tokenizer.ggml.scores"),
        Some(vec![1.5, 0.5, -1.0])
    );

    let names = g.names();
    assert_eq!(names.len(), 3);

    let f32 = g.read("t_f32").unwrap();
    assert_eq!(f32.shape, vec![2, 16]);
    assert_eq!(f32.data.into_f32().unwrap(), f32_data);

    let bf16 = g.read("t_bf16").unwrap();
    assert_eq!(bf16.shape, vec![2, 32]);
    let got: Vec<f32> = bf16
        .data
        .into_f32()
        .unwrap()
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
    let got = q8.data.into_f32().unwrap();
    for (a, b) in got.iter().zip(&q8_data) {
        assert!((a - b).abs() < 0.5, "{a} vs {b}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

fn synth_qk_gguf(arch: &str, interleaved: bool) -> Vec<u8> {
    const HD: u32 = 4;
    const HEADS: u32 = 2;
    const ROWS: u32 = HEADS * HD;
    const COLS: u32 = 6;
    let mut h = Vec::new();
    w_u32(&mut h, 0x4655_4747);
    w_u32(&mut h, 3);
    w_u64(&mut h, 2);
    w_u64(&mut h, 4);
    w_kv_str(&mut h, "general.architecture", arch);
    w_kv_u32(&mut h, "general.alignment", 64);
    w_kv_u32(&mut h, "llama.block_count", 1);
    w_kv_u32(&mut h, "llama.attention.key_length", HD);
    w_str(&mut h, "blk.0.attn_q.weight");
    w_u32(&mut h, 2);
    w_u64(&mut h, COLS as u64);
    w_u64(&mut h, ROWS as u64);
    w_u32(&mut h, 0);
    w_u64(&mut h, 0);
    w_str(&mut h, "blk.0.attn_q.bias");
    w_u32(&mut h, 1);
    w_u64(&mut h, ROWS as u64);
    w_u32(&mut h, 0);
    w_u64(&mut h, ROWS as u64 * COLS as u64 * 4);
    while h.len() % 64 != 0 {
        h.push(0);
    }
    let mut rows = (0..ROWS).collect::<Vec<_>>();
    if interleaved {
        for hd_i in 0..HEADS {
            let base = hd_i * HD;
            let r = &mut rows[base as usize..(base + HD) as usize];
            let mut p = vec![0u32; HD as usize];
            for i in 0..HD as usize / 2 {
                p[i] = r[2 * i];
                p[HD as usize / 2 + i] = r[2 * i + 1];
            }
            r.copy_from_slice(&p);
        }
    }
    for &r in &rows {
        for c in 0..COLS {
            h.extend_from_slice(&((r * 1000 + c) as f32).to_le_bytes());
        }
    }
    for &r in &rows {
        h.extend_from_slice(&((r + 1000) as f32).to_le_bytes());
    }
    h
}

fn open_bytes(tag: &str, b: &[u8]) -> Gguf {
    let dir = tmp_dir(&format!("qk-{tag}"));
    let path = dir.join("qk.gguf");
    std::fs::write(&path, b).unwrap();
    let g = Gguf::open(&path).unwrap();
    std::fs::remove_dir_all(&dir).ok();
    g
}

#[test]
fn deinterleaves_llama_qk_rows() {
    let g = open_bytes("llama", &synth_qk_gguf("llama", true));
    let w = g.read("blk.0.attn_q.weight").unwrap();
    let expect: Vec<f32> = (0..8u32)
        .flat_map(|r| (0..6u32).map(move |c| (r * 1000 + c) as f32))
        .collect();
    assert_eq!(
        w.data.into_f32().unwrap(),
        expect,
        "llama q weight restored to plain order"
    );
    let b = g.read("blk.0.attn_q.bias").unwrap();
    let expect_bias: Vec<f32> = (0..8u32).map(|r| (r + 1000) as f32).collect();
    assert_eq!(
        b.data.into_f32().unwrap(),
        expect_bias,
        "llama q bias restored to plain order"
    );
}

#[test]
fn leaves_non_llama_qk_rows_untouched() {
    for arch in ["qwen2", "gemma3", "llama4"] {
        let g = open_bytes(arch, &synth_qk_gguf(arch, true));
        let w = g.read("blk.0.attn_q.weight").unwrap();
        let got = w.data.into_f32().unwrap();
        assert_ne!(
            got,
            (0..8u32)
                .flat_map(|r| (0..6u32).map(move |c| (r * 1000 + c) as f32))
                .collect::<Vec<_>>(),
            "{arch} q must not be deinterleaved"
        );
    }
}



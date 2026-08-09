use flint_onnx::tensor::Data;
use flint_onnx::{Session, Tensor};

fn model_path() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("ONNX_TEST_MODEL") {
        return Some(std::path::PathBuf::from(p));
    }
    let p = std::path::Path::new("temp/onnx-test/model.onnx");
    p.exists().then(|| p.to_path_buf())
}

const REF_FIRST: [f32; 8] = [
    0.05258789, -0.07846259, 0.25362775, -0.19136892, -0.59708065, -0.06743392, 0.06855663,
    -0.19574150,
];

#[test]
fn minilm_end_to_end() {
    let Some(path) = model_path() else {
        eprintln!("skipping: model not found (set ONNX_TEST_MODEL)");
        return;
    };
    let mut s = Session::load(&path).unwrap();
    let ids: Vec<i64> = vec![101, 7592, 2087, 2023, 2003, 1037, 3710, 102];
    s.set_input("input_ids", Tensor::i64(ids, vec![1, 8])).unwrap();
    s.set_input("attention_mask", Tensor::i64(vec![1; 8], vec![1, 8])).unwrap();
    s.set_input("token_type_ids", Tensor::i64(vec![0; 8], vec![1, 8])).unwrap();
    let out = s.run().unwrap();
    let t = &out["last_hidden_state"];
    assert_eq!(t.shape, vec![1, 8, 384]);
    let Data::F32(v) = &t.data else {
        panic!("output not f32");
    };
    assert_eq!(v.len(), 8 * 384);

    for (i, (a, b)) in v.iter().take(8).zip(REF_FIRST).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "element {i}: flint {a} vs reference {b}"
        );
    }

    assert!(v.iter().all(|x| x.is_finite() && x.abs() < 100.0));
}

#[test]
fn minilm_matches_fp16_and_int8_variants() {

    let Some(base) = model_path() else {
        eprintln!("skipping: model not found");
        return;
    };
    let dir = base.parent().unwrap();
    let variants = [
        ("model_fp16.onnx", 5e-2),
        ("model_quantized.onnx", 5e-1),
    ];
    let mut s = Session::load(&base).unwrap();
    let ids: Vec<i64> = vec![101, 7592, 2087, 2023, 2003, 1037, 3710, 102];
    s.set_input("input_ids", Tensor::i64(ids.clone(), vec![1, 8])).unwrap();
    s.set_input("attention_mask", Tensor::i64(vec![1; 8], vec![1, 8])).unwrap();
    s.set_input("token_type_ids", Tensor::i64(vec![0; 8], vec![1, 8])).unwrap();
    let base_out = s.run().unwrap();
    let Data::F32(base_v) = &base_out["last_hidden_state"].data else {
        panic!("output not f32");
    };
    for (file, tol) in variants {
        let p = dir.join(file);
        if !p.exists() {
            eprintln!("skipping {file}: not present");
            continue;
        }
        let mut s = Session::load(&p).unwrap();
        s.set_input("input_ids", Tensor::i64(ids.clone(), vec![1, 8])).unwrap();
        s.set_input("attention_mask", Tensor::i64(vec![1; 8], vec![1, 8])).unwrap();
        s.set_input("token_type_ids", Tensor::i64(vec![0; 8], vec![1, 8])).unwrap();
        let out = s.run().unwrap();
        let Data::F32(v) = &out["last_hidden_state"].data else {
            panic!("{file} output not f32");
        };
        let max_err = v
            .iter()
            .zip(base_v)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < tol,
            "{file} deviates from fp32 by {max_err} (tol {tol})"
        );
    }
}

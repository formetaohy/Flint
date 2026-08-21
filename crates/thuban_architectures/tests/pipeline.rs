use thuban_backend::Backend;
use thuban_checkpoint::GgufWriter;

#[test]
fn unsupported_formats_fail_fast() {
    let backend = Backend::new().unwrap();

    let dir = std::env::temp_dir().join(format!("thuban-badarch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut w = GgufWriter::new(32);
    w.kv_str("general.architecture", "bert");
    std::fs::write(dir.join("model.gguf"), w.finish()).unwrap();
    let err = thuban_architectures::load(
        &dir,
        &thuban_architectures::LoadOptions {
            seq_lens: vec![64],
            pages: None,
            spec_depth: None,
        },
        &backend,
    )
    .err()
    .unwrap();
    assert!(
        err.to_string().contains("unsupported GGUF architecture"),
        "{err}"
    );

    let gguf_dir = std::env::temp_dir().join(format!("thuban-badarch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&gguf_dir);
    std::fs::create_dir_all(&gguf_dir).unwrap();
    let mut w = GgufWriter::new(32);
    w.kv_u32("llama.block_count", 1);
    std::fs::write(gguf_dir.join("model.gguf"), w.finish()).unwrap();
    let err = thuban_architectures::load(
        &gguf_dir,
        &thuban_architectures::LoadOptions {
            seq_lens: vec![64],
            pages: None,
            spec_depth: None,
        },
        &backend,
    )
    .err()
    .unwrap();
    assert!(
        err.to_string().contains("missing general.architecture"),
        "{err}"
    );
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&gguf_dir).ok();
}

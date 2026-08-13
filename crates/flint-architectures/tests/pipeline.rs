use flint_backend::Backend;
use flint_checkpoint::{GgufWriter, SafetensorEntry};

#[test]
fn unsupported_formats_fail_fast() {
    let backend = Backend::new().unwrap();

    let dir = std::env::temp_dir().join(format!("flint-badfmt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    flint_checkpoint::write_tensors(
        &dir.join("model.safetensors"),
        &[SafetensorEntry {
            name: "a",
            shape: &[1],
            bytes: &[0u8; 4],
            bf16: false,
        }],
    )
    .unwrap();
    std::fs::write(dir.join("config.json"), r#"{"model_type": "bert"}"#).unwrap();
    let err = flint_architectures::load(&dir, 64, &backend).err().unwrap();
    assert!(err.to_string().contains("unsupported model_type"), "{err}");

    let gguf_dir = std::env::temp_dir().join(format!("flint-badarch-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&gguf_dir);
    std::fs::create_dir_all(&gguf_dir).unwrap();
    let mut w = GgufWriter::new(32);
    w.kv_u32("llama.block_count", 1);
    std::fs::write(gguf_dir.join("model.gguf"), w.finish()).unwrap();
    let err = flint_architectures::load(&gguf_dir, 64, &backend)
        .err()
        .unwrap();
    assert!(
        err.to_string().contains("missing general.architecture"),
        "{err}"
    );
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&gguf_dir).ok();
}

use std::path::Path;

use flint_backend::Backend;
use flint_error::Result;

pub fn run(dir: &Path, prompt: &str) -> Result<()> {
    let tokenizer_file = dir.join("tokenizer.json");
    if !tokenizer_file.exists() {
        return Err(flint_error::Error::Model(format!(
            "embedding needs {} in the repo",
            tokenizer_file.display()
        )));
    }
    let tokenizer = flint_tokenizer::Tokenizer::from_file(&tokenizer_file)?;
    let ids = tokenizer.encode(prompt)?;
    if ids.is_empty() {
        return Err(flint_error::Error::Tokenizer("empty prompt".into()));
    }

    eprintln!("[flint] initializing GPU backend...");
    let mut backend = Backend::new()?;
    eprintln!("[flint] adapter: {}", backend.adapter_name());

    eprintln!("[flint] loading embedder from {}...", dir.display());
    let load_t = std::time::Instant::now();
    let mut embedder = flint_architectures::load_embedder(dir, &backend)?;
    eprintln!("[flint] loaded in {:.1}s", load_t.elapsed().as_secs_f64());

    let t0 = std::time::Instant::now();
    let embedding = embedder.embed(&mut backend, &ids)?;
    eprintln!("[flint] embedded in {:.2}s", t0.elapsed().as_secs_f64());
    let show: Vec<String> = embedding
        .iter()
        .take(8)
        .map(|v| format!("{v:.4}"))
        .collect();
    println!(
        "embedding: [{} ...] ({} dims)",
        show.join(", "),
        embedding.len()
    );
    Ok(())
}

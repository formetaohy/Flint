use std::io::Write as _;
use std::path::Path;

use flint_backend::Backend;
use flint_error::Result;
use flint_generate::{Engine, Sampler};

const SYSTEM: &str = "You are a helpful assistant.";

pub fn run(dir: &Path, prompt: &str, max_tokens: usize, ctx_size: u32) -> Result<()> {
    eprintln!("[flint] initializing GPU backend...");
    let backend = Backend::new()?;
    eprintln!("[flint] adapter: {}", backend.adapter_name());

    eprintln!("[flint] loading model from {}...", dir.display());
    let load_t = std::time::Instant::now();
    let chat_model = flint_architectures::load(dir, ctx_size, &backend)?;
    eprintln!("[flint] loaded in {:.1}s", load_t.elapsed().as_secs_f64());

    let sampler = Sampler::new(Default::default(), 42);
    let mut engine = Engine::new(
        backend,
        chat_model.model,
        chat_model.tokenizer,
        sampler,
        chat_model.stop,
        false,
    );
    let text = chat_model.chat.render(SYSTEM, &[], prompt);
    let mut stream = engine.stream(&text, max_tokens)?;
    for piece in stream.by_ref() {
        let piece = piece?;
        print!("{}", piece.text);
        std::io::stdout().flush().ok();
    }
    println!();
    eprintln!("{}", stream.stats().summary());
    Ok(())
}

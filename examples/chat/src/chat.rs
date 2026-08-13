use std::io::Write as _;
use std::path::Path;

use flint_backend::Backend;
use flint_error::Result;
use flint_generate::{Engine, GenStats, Sampler};

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
    eprintln!("{}", stats_summary(stream.stats()));
    Ok(())
}

fn stats_summary(s: &GenStats) -> String {
    let pp = if s.prefill_secs > 0.0 {
        s.prefill_tokens as f64 / s.prefill_secs
    } else {
        0.0
    };
    let tg = if s.decode_secs > 0.0 {
        s.decode_tokens as f64 / s.decode_secs
    } else {
        0.0
    };
    format!(
        "[flint] prefill: {} tok in {:.2}s ({pp:.1} tok/s) | decode: {} tok in {:.2}s ({tg:.1} tok/s)",
        s.prefill_tokens, s.prefill_secs, s.decode_tokens, s.decode_secs,
    )
}

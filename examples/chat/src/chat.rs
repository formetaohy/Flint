use std::io::Write as _;
use std::path::Path;

use flint_backend::Backend;
use flint_error::Result;
use flint_generate::{Engine, GenStats};

pub fn run(dir: &Path, prompt: &str, max_tokens: usize, ctx_size: u32) -> Result<()> {
    eprintln!("[flint] initializing GPU backend...");
    let backend = Backend::new()?;
    eprintln!("[flint] adapter: {}", backend.adapter_name());

    eprintln!("[flint] loading model from {}...", dir.display());
    let load_t = std::time::Instant::now();
    let chat_model = flint_architectures::load(
        dir,
        &flint_architectures::LoadOptions {
            slots: vec![ctx_size],
            spec_depth: None,
        },
        &backend,
    )?;
    eprintln!("[flint] loaded in {:.1}s", load_t.elapsed().as_secs_f64());

    let mut engine = Engine::new(
        backend,
        chat_model.model,
        chat_model.tokenizer,
        Default::default(),
        42,
        chat_model.stop,
        false,
    );
    let text = chat_model.chat.render(SYSTEM, &[], prompt);
    let session = engine.create(&text, max_tokens, None)?;
    loop {
        engine.step()?;
        for piece in engine.poll(session) {
            print!("{}", piece.text);
            std::io::stdout().flush().ok();
        }
        if engine.finished(session) {
            break;
        }
    }
    println!();
    if let Some(s) = engine.stats(session) {
        eprintln!("{}", stats_summary(&s));
    }
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

const SYSTEM: &str = "You are a helpful assistant.";

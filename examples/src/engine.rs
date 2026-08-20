use std::io::Write as _;
use std::path::Path;

use flint_architectures::LoadOptions;
use flint_backend::Backend;
use flint_error::Result;
use flint_generate::{Engine, GenStats, Grammar, SessionId};

pub fn spawn(
    dir: &Path,
    system: &str,
    prompt: &str,
    max_tokens: usize,
    ctx_size: u32,
    grammar: Option<Grammar>,
) -> Result<(Engine, SessionId)> {
    eprintln!("[flint] initializing GPU backend...");
    let backend = Backend::new()?;
    eprintln!("[flint] adapter: {}", backend.adapter_name());

    eprintln!("[flint] loading model from {}...", dir.display());
    let load_t = std::time::Instant::now();
    let chat_model = flint_architectures::load(
        dir,
        &LoadOptions {
            seq_lens: vec![ctx_size],
            pages: None,
            spec_depth: None,
        },
        &backend,
    )?;
    eprintln!("[flint] loaded in {:.1}s", load_t.elapsed().as_secs_f64());

    let text = chat_model.chat.render(system, &[], prompt);
    let mut engine = Engine::new(
        backend,
        chat_model.model,
        chat_model.tokenizer,
        Default::default(),
        42,
        chat_model.stop,
        false,
    );
    let id = engine.create(&text, max_tokens, grammar)?;
    Ok((engine, id))
}

pub fn stream(engine: &mut Engine, id: SessionId) -> Result<()> {
    loop {
        engine.step()?;
        for piece in engine.poll(id) {
            print!("{}", piece.text);
            std::io::stdout().flush().ok();
        }
        if engine.finished(id) {
            break;
        }
    }
    println!();
    if let Some(s) = engine.stats(id) {
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

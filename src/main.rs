use std::io::{BufRead, Write as _};
use std::path::PathBuf;

use clap::Parser;

use flint_architectures::ChatFormat;
use flint_backend::Backend;
use flint_error::Result;
use flint_generate::{Engine, Sampler, SamplingParams};

/// Flint: a local LLM inference engine on WGPU.
#[derive(Parser)]
#[command(name = "flint", version, about)]
struct Args {
    /// Directory containing config.json / tokenizer.json / safetensors shards.
    #[arg(long)]
    model: PathBuf,

    /// One-shot prompt; omit for an interactive session.
    #[arg(long)]
    prompt: Option<String>,

    #[arg(long, default_value = "You are a helpful assistant.")]
    system: String,

    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    #[arg(long, default_value_t = 0.8)]
    top_p: f32,

    #[arg(long, default_value_t = 20)]
    top_k: usize,

    /// Drop tokens whose probability is below min_p * max_prob.
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,

    /// Multiplicative repetition penalty (1.0 disables).
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Context window (also bounds the KV cache allocation).
    #[arg(long, default_value_t = 4096)]
    max_seq: u32,

    /// Enable MTP speculative decoding (correct but only faster on
    /// bandwidth-rich adapters; the draft head shares the vocab projection).
    #[arg(long)]
    speculate: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    eprintln!("[flint] initializing GPU backend...");
    let backend = Backend::new()?;
    eprintln!("[flint] adapter: {}", backend.adapter_name());

    eprintln!("[flint] loading weights from {}...", args.model.display());
    let load_t = std::time::Instant::now();
    let chat_model = flint_architectures::load(&args.model, args.max_seq, &backend)?;
    eprintln!(
        "[flint] weights loaded in {:.1}s",
        load_t.elapsed().as_secs_f64()
    );

    let sampler = Sampler::new(
        SamplingParams {
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            min_p: args.min_p,
            repeat_penalty: args.repeat_penalty,
            ..Default::default()
        },
        args.seed,
    );
    let mut engine = Engine::new(
        backend,
        chat_model.model,
        chat_model.tokenizer,
        sampler,
        chat_model.stop,
        args.speculate,
    );
    let chat = chat_model.chat;

    match args.prompt {
        Some(prompt) => {
            run_turn(
                &mut engine,
                chat.as_ref(),
                &args.system,
                &[],
                &prompt,
                args.max_tokens,
            )?;
        }
        None => interactive(&mut engine, chat.as_ref(), &args.system, args.max_tokens)?,
    }
    if let Some(report) = engine.profile_report() {
        eprint!("{report}");
    }
    Ok(())
}

/// Interactive REPL: reads user turns, keeps them as history, and streams each
/// assistant reply back to stdout.
fn interactive(
    engine: &mut Engine,
    chat: &dyn ChatFormat,
    system: &str,
    max_tokens: usize,
) -> Result<()> {
    eprintln!("[flint] interactive mode. type 'exit' to quit.");
    let mut history: Vec<(String, String)> = Vec::new();
    let stdin = std::io::stdin();
    loop {
        print!("> ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let user = line.trim().to_string();
        if user.is_empty() {
            continue;
        }
        if user == "exit" {
            break;
        }
        let reply = run_turn(engine, chat, system, &history, &user, max_tokens)?;
        history.push((user, reply));
    }
    Ok(())
}

/// Generates one assistant turn, streaming pieces to stdout. Returns the
/// assembled reply text.
fn run_turn(
    engine: &mut Engine,
    chat: &dyn ChatFormat,
    system: &str,
    history: &[(String, String)],
    user: &str,
    max_tokens: usize,
) -> Result<String> {
    let text = chat.render(system, history, user);
    let mut stream = engine.stream(&text, max_tokens)?;
    let mut reply = String::new();
    for piece in stream.by_ref() {
        let piece = piece?;
        print!("{}", piece.text);
        std::io::stdout().flush().ok();
        reply.push_str(&piece.text);
    }
    println!();
    eprintln!("{}", stream.stats().summary());
    Ok(reply)
}

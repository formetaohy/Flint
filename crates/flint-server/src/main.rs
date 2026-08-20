use std::path::PathBuf;

use clap::Parser;
use flint_error::Result;
use flint_fetch::{Repo, fetch};

use flint_server::hub::Hub;
use flint_server::server::{ServerConfig, serve};

#[derive(Parser)]
#[command(
    name = "flint-server",
    version,
    about = "serve a local model behind OpenAI, Anthropic and Gemini compatible APIs"
)]
struct Args {
    #[arg(long)]
    model: Option<String>,


    #[arg(long)]
    dir: Option<PathBuf>,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 8080)]
    port: u16,

    #[arg(long)]
    api_key: Option<String>,

    #[arg(long, default_value_t = 32768)]
    ctx_size: u32,

    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long)]
    speculate: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let dir = match (&args.model, &args.dir) {
        (Some(model), None) => {
            let dir = PathBuf::from("temp").join(model.replace('/', "--"));
            fetch(&Repo::new(model), &dir)?;
            dir
        }
        (None, Some(dir)) => dir.clone(),
        _ => {
            eprintln!("pass exactly one of --model (Hugging Face repo) or --dir (local path)");
            std::process::exit(2);
        }
    };
    let model_id = args
        .model
        .clone()
        .unwrap_or_else(|| dir.display().to_string());

    eprintln!("[flint] initializing GPU backend...");
    let backend = flint_backend::Backend::new()?;
    eprintln!("[flint] adapter: {}", backend.adapter_name());

    eprintln!("[flint] loading model from {}...", dir.display());
    let load_t = std::time::Instant::now();
    let chat_model = flint_architectures::load(
        &dir,
        &flint_architectures::LoadOptions {
            seq_lens: vec![args.ctx_size],
            pages: None,
            spec_depth: None,
        },
        &backend,
    )?;
    eprintln!("[flint] loaded in {:.1}s", load_t.elapsed().as_secs_f64());

    let engine = flint_generate::Engine::new(
        backend,
        chat_model.model,
        chat_model.tokenizer.clone(),
        Default::default(),
        args.seed,
        chat_model.stop,
        args.speculate,
    );
    let hub = Hub::new(
        engine,
        chat_model.chat,
        chat_model.tokenizer,
        model_id,
        args.ctx_size,
        args.max_tokens,
    );
    serve(
        ServerConfig {
            host: args.host,
            port: args.port,
            api_key: args.api_key,
        },
        hub,
    )
}

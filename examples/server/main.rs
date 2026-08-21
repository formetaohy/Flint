use std::path::PathBuf;

use clap::Parser;
use thuban_error::{Error, Result};
use thuban_fetch::{Repo, fetch};
use thuban_server::bootstrap;

#[derive(Parser)]
#[command(
    name = "server",
    version,
    about = "serve a downloaded Hugging Face model behind OpenAI, Anthropic and Gemini compatible APIs"
)]
struct Args {
    #[arg(long)]
    model: String,

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
    let dir = PathBuf::from("temp").join(args.model.replace('/', "--"));
    fetch(&Repo::new(&args.model), &dir).map_err(|e| {
        Error::Config(format!("download {}: {e}", args.model))
    })?;
    bootstrap::serve_from(bootstrap::ServeOptions {
        dir,
        model_id: args.model,
        host: args.host,
        port: args.port,
        api_key: args.api_key,
        ctx_size: args.ctx_size,
        max_tokens: args.max_tokens,
        seed: args.seed,
        speculate: args.speculate,
    })
}

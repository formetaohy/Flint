use std::path::PathBuf;

use clap::Parser;
use thuban_error::{Error, Result};

use thuban_server::bootstrap;

#[derive(Parser)]
#[command(
    name = "thuban-server",
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
    let dir = match bootstrap::model_dir(args.model.as_deref(), args.dir.as_deref())? {
        Some(dir) => dir,
        None => {
            return Err(Error::Config(
                "pass exactly one of --model (Hugging Face repo) or --dir (local path)".into(),
            ));
        }
    };
    let model_id = args
        .model
        .unwrap_or_else(|| dir.display().to_string());
    bootstrap::serve_from(bootstrap::ServeOptions {
        dir,
        model_id,
        host: args.host,
        port: args.port,
        api_key: args.api_key,
        ctx_size: args.ctx_size,
        max_tokens: args.max_tokens,
        seed: args.seed,
        speculate: args.speculate,
    })
}

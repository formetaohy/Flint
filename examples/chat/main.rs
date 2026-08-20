use std::path::PathBuf;

use clap::Parser;
use flint_error::Result;
use flint_examples::assets::{self, Format};
use flint_examples::engine;
use flint_examples::hub::Hub;

#[derive(Parser)]
#[command(
    name = "chat",
    version,
    about = "download a Hugging Face model into temp/ and run chat inference"
)]
struct Args {
    #[arg(long)]
    model: String,

    #[arg(long)]
    prompt: String,

    #[arg(long, value_enum, default_value_t = Format::Gguf)]
    format: Format,

    #[arg(long, default_value_t = 8192)]
    max_tokens: usize,

    #[arg(long, default_value_t = 32768)]
    ctx_size: u32,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let dir = PathBuf::from("temp").join(args.model.replace('/', "--"));
    assets::ensure(&Hub::new(&args.model), args.format, &dir)?;
    let (mut engine, id) = engine::spawn(
        &dir,
        SYSTEM,
        &args.prompt,
        args.max_tokens,
        args.ctx_size,
        None,
    )?;
    engine::stream(&mut engine, id)
}

const SYSTEM: &str = "You are a helpful assistant.";

use std::path::PathBuf;

use clap::Parser;
use thuban_error::Result;
use thuban_examples::engine;
use thuban_fetch::{Repo, fetch};

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

    #[arg(long, default_value_t = 8192)]
    max_tokens: usize,

    #[arg(long, default_value_t = 32768)]
    ctx_size: u32,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let dir = PathBuf::from("temp").join(args.model.replace('/', "--"));
    fetch(&Repo::new(&args.model), &dir)?;
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

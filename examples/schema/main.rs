use std::path::PathBuf;

use clap::Parser;
use thuban_error::{Error, Result};
use thuban_examples::engine;
use thuban_generate::Grammar;
use thuban_fetch::{Repo, fetch};

#[derive(Parser)]
#[command(
    name = "schema",
    version,
    about = "download a Hugging Face model into temp/ and generate JSON conforming to a JSON Schema"
)]
struct Args {
    #[arg(long)]
    model: String,

    #[arg(long)]
    prompt: String,

    #[arg(long)]
    schema: PathBuf,

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
    let text = std::fs::read_to_string(&args.schema)
        .map_err(|e| Error::Config(format!("read {}: {e}", args.schema.display())))?;
    let schema = serde_json::from_str(&text)
        .map_err(|e| Error::Config(format!("parse {}: {e}", args.schema.display())))?;
    let grammar = Grammar::from_schema(&schema)?;
    let (mut engine, id) = engine::spawn(
        &dir,
        SYSTEM,
        &args.prompt,
        args.max_tokens,
        args.ctx_size,
        Some(grammar),
    )?;
    engine::stream(&mut engine, id)
}

const SYSTEM: &str = "You are a helpful assistant that always answers in valid JSON.";

use std::io::Write as _;
use std::path::PathBuf;

use clap::Parser;
use thuban_error::{Error, Result};
use thuban_examples::engine;

#[derive(Parser)]
#[command(name = "verify", about = "run inference on a single local .gguf file")]
struct Args {
    #[arg(long)]
    gguf: PathBuf,

    #[arg(long, default_value = "What is 2+2? Answer briefly.")]
    prompt: String,

    #[arg(long, default_value_t = 64)]
    max_tokens: usize,

    #[arg(long, default_value_t = 4096)]
    ctx_size: u32,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    if !args.gguf.is_file() {
        return Err(Error::Model(format!(
            "{} is not a file",
            args.gguf.display()
        )));
    }
    let dir = PathBuf::from("temp")
        .join("verify")
        .join(format!("run-{}", std::process::id()));
    std::fs::create_dir_all(&dir)
        .map_err(|e| Error::Model(format!("mkdir {}: {e}", dir.display())))?;
    let local = dir.join(args.gguf.file_name().expect("gguf file name"));
    std::fs::hard_link(&args.gguf, &local).map_err(|e| {
        Error::Model(format!(
            "hard link {} -> {}: {e}",
            args.gguf.display(),
            local.display()
        ))
    })?;
    let run = engine::spawn(
        &dir,
        SYSTEM,
        &args.prompt,
        args.max_tokens,
        args.ctx_size,
        None,
    );
    let result = match run {
        Ok((mut engine, id)) => {
            if std::env::var("VERIFY_IDS").is_ok() {
                loop {
                    engine.step()?;
                    for piece in engine.poll(id) {
                        print!("{}[{}]", piece.text, piece.token);
                        std::io::stdout().flush().ok();
                    }
                    if engine.finished(id) {
                        break;
                    }
                }
                println!();
                Ok(())
            } else {
                engine::stream(&mut engine, id)
            }
        }
        Err(e) => Err(e),
    };
    let _ = std::fs::remove_dir_all(&dir);
    result
}

const SYSTEM: &str = "You are a helpful assistant.";

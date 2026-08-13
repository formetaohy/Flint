mod chat;
mod hub;
mod onnx;

use std::fs;
use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};

use flint_error::{Error, Result};

use hub::{FileEntry, Hub};

#[derive(Parser)]
#[command(
    name = "chat",
    version,
    about = "download a Hugging Face model into temp/ and run inference"
)]
struct Args {
    #[arg(long)]
    model: String,

    #[arg(long)]
    prompt: String,

    #[arg(long, value_enum)]
    format: Format,

    #[arg(long, default_value_t = 8192)]
    max_tokens: usize,

    #[arg(long, default_value_t = 32768)]
    ctx_size: u32,
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    #[value(alias = "safetensor")]
    Safetensors,
    Gguf,
    Onnx,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let dir = PathBuf::from("temp").join(args.model.replace('/', "--"));
    let assets = ensure_assets(&Hub::new(&args.model), args.format, &dir)?;
    match args.format {
        Format::Onnx => {
            let model_file = assets
                .iter()
                .find(|p| p.extension().is_some_and(|x| x == "onnx"))
                .ok_or_else(|| Error::Model(format!("no .onnx file in {}", dir.display())))?;
            onnx::run(model_file, &dir, &args.prompt)
        }
        Format::Safetensors | Format::Gguf => {
            chat::run(&dir, &args.prompt, args.max_tokens, args.ctx_size)
        }
    }
}

fn ensure_assets(hub: &Hub, format: Format, dir: &Path) -> Result<Vec<PathBuf>> {
    let files = hub.files()?;
    let wanted = select(format, &files)?;
    fs::create_dir_all(dir).map_err(|e| Error::Model(format!("mkdir {}: {e}", dir.display())))?;
    let mut local = vec![];
    for entry in wanted {
        let dest = dir.join(&entry.path);
        let complete = dest
            .metadata()
            .map(|m| m.is_file() && m.len() == entry.size)
            .unwrap_or(false);
        if complete {
            eprintln!("[hf] {} present, skipping", entry.path);
        } else {
            let written = hub.download(&entry.path, &dest)?;
            if written != entry.size {
                return Err(Error::Model(format!(
                    "{}: downloaded {written} bytes, expected {}",
                    entry.path, entry.size
                )));
            }
        }
        local.push(dest);
    }
    Ok(local)
}

fn select(format: Format, files: &[FileEntry]) -> Result<Vec<FileEntry>> {
    match format {
        Format::Safetensors => select_safetensors(files),
        Format::Gguf => select_single(files, "gguf"),
        Format::Onnx => select_single(files, "onnx"),
    }
}

fn select_safetensors(files: &[FileEntry]) -> Result<Vec<FileEntry>> {
    let mut wanted = vec![];
    for name in [
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ] {
        if let Some(entry) = files.iter().find(|f| f.path == name) {
            wanted.push(entry.clone());
        }
    }
    for entry in files.iter().filter(|f| f.path.ends_with(".safetensors")) {
        wanted.push(entry.clone());
    }
    let has = |name: &str| wanted.iter().any(|f| f.path == name);
    if !has("config.json") {
        return Err(Error::Model("repo has no config.json".into()));
    }
    if !has("tokenizer.json") {
        return Err(Error::Model("repo has no tokenizer.json".into()));
    }
    if !wanted.iter().any(|f| f.path.ends_with(".safetensors")) {
        return Err(Error::Model("repo has no .safetensors weights".into()));
    }
    Ok(wanted)
}

fn select_single(files: &[FileEntry], ext: &str) -> Result<Vec<FileEntry>> {
    let mut matches: Vec<FileEntry> = files
        .iter()
        .filter(|f| f.path.ends_with(&format!(".{ext}")))
        .cloned()
        .collect();
    if matches.is_empty() {
        return Err(Error::Model(format!("repo has no .{ext} file")));
    }
    matches.sort_by_key(|f| std::cmp::Reverse(f.size));
    if matches.len() > 1 {
        eprintln!(
            "[hf] {} .{ext} files, using largest: {} ({:.1} MiB)",
            matches.len(),
            matches[0].path,
            matches[0].size as f64 / (1 << 20) as f64
        );
    }
    let mut wanted = vec![matches.remove(0)];
    if let Some(entry) = files.iter().find(|f| f.path == "tokenizer.json") {
        wanted.push(entry.clone());
    }
    Ok(wanted)
}

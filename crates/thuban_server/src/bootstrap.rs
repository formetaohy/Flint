use std::path::{Path, PathBuf};

use thuban_architectures::LoadOptions;
use thuban_backend::Backend;
use thuban_error::Result;
use thuban_generate::Engine;

use crate::generator::Generator;
use crate::server::{ServerConfig, serve};

pub struct ServeOptions {
    pub dir: PathBuf,
    pub model_id: String,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub ctx_size: u32,
    pub max_tokens: usize,
    pub seed: u64,
    pub speculate: bool,
}

pub fn serve_from(opts: ServeOptions) -> Result<()> {
    let generator = load_generator(&opts)?;
    serve(
        ServerConfig {
            host: opts.host,
            port: opts.port,
            api_key: opts.api_key,
        },
        generator,
    )
}

pub fn load_generator(opts: &ServeOptions) -> Result<Generator> {
    eprintln!("[thuban] initializing GPU backend...");
    let backend = Backend::new()?;
    eprintln!("[thuban] adapter: {}", backend.adapter_name());

    eprintln!("[thuban] loading model from {}...", opts.dir.display());
    let load_t = std::time::Instant::now();
    let chat_model = thuban_architectures::load(
        &opts.dir,
        &LoadOptions {
            seq_lens: vec![opts.ctx_size],
            pages: None,
            spec_depth: None,
        },
        &backend,
    )?;
    eprintln!("[thuban] loaded in {:.1}s", load_t.elapsed().as_secs_f64());

    let engine = Engine::new(
        backend,
        chat_model.model,
        chat_model.tokenizer.clone(),
        Default::default(),
        opts.seed,
        chat_model.stop,
        opts.speculate,
    );
    Ok(Generator::new(
        engine,
        chat_model.chat,
        chat_model.tokenizer,
        opts.model_id.clone(),
        opts.ctx_size,
        opts.max_tokens,
    ))
}

pub fn model_dir(model: Option<&str>, dir: Option<&Path>) -> Result<Option<PathBuf>> {
    match (model, dir) {
        (Some(model), None) => {
            let dir = PathBuf::from("temp").join(model.replace('/', "--"));
            thuban_fetch::fetch(&thuban_fetch::Repo::new(model), &dir)?;
            Ok(Some(dir))
        }
        (None, Some(dir)) => Ok(Some(dir.to_path_buf())),
        _ => Ok(None),
    }
}

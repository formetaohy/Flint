use std::io::Write as _;
use std::path::{Path, PathBuf};

use clap::{Args as ClapArgs, Parser};

use flint_architectures::chat::ChatFormat;
use flint_backend::Backend;
use flint_error::Result;
use flint_generate::{Engine, Sampler, SamplingParams};

#[derive(Parser)]
#[command(name = "flint", version, about)]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[command(flatten)]
    onnx: OnnxArgs,

    #[command(flatten)]
    chat: ChatArgs,
}

#[derive(ClapArgs)]
struct OnnxArgs {
    #[arg(long)]
    inputs: Option<String>,

    #[arg(long)]
    inputs_file: Option<PathBuf>,

    #[arg(long)]
    output: Vec<String>,

    #[arg(long)]
    full: bool,
}

impl OnnxArgs {
    fn is_empty(&self) -> bool {
        self.inputs.is_none() && self.inputs_file.is_none() && self.output.is_empty() && !self.full
    }
}

#[derive(ClapArgs)]
struct ChatArgs {
    #[arg(long)]
    prompt: Option<String>,

    #[arg(long, default_value = "You are a helpful assistant.")]
    system: String,

    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,

    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    #[arg(long, default_value_t = 40)]
    top_k: usize,

    #[arg(long, default_value_t = 0.05)]
    min_p: f32,

    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 8192)]
    ctx_size: u32,

    #[arg(long)]
    speculate: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    if is_onnx(&args.model) {
        onnx_run(&args.model, args.onnx)
    } else {
        if !args.onnx.is_empty() {
            return Err(flint_error::Error::Model(
                "onnx options --inputs/--inputs-file/--output/--full require an .onnx model".into(),
            ));
        }
        chat_main(&args.model, args.chat)
    }
}

fn is_onnx(model: &Path) -> bool {
    model
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
}

fn onnx_run(model: &Path, args: OnnxArgs) -> Result<()> {
    let json = match (args.inputs, args.inputs_file) {
        (Some(_), Some(_)) => {
            return Err(flint_error::Error::Model(
                "provide either --inputs or --inputs-file, not both".into(),
            ));
        }
        (Some(s), None) => s,
        (None, Some(f)) => std::fs::read_to_string(&f)
            .map_err(|e| flint_error::Error::Model(format!("cannot read {}: {e}", f.display())))?,
        (None, None) => {
            return Err(flint_error::Error::Model(
                "onnx model requires --inputs or --inputs-file".into(),
            ));
        }
    };
    let inputs: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&json)
        .map_err(|e| flint_error::Error::Model(format!("invalid inputs JSON: {e}")))?;

    let mut session = flint_onnx::Session::load(model)?;
    for (name, value) in &inputs {
        let t = json_to_tensor(value)?;
        eprintln!("[flint] input {name}: shape {:?}", t.shape);
        session.set_input(name, t)?;
    }

    for v in &session.graph().inputs {
        if !inputs.contains_key(&v.name) {
            return Err(flint_error::Error::Model(format!(
                "graph input {:?} is not provided",
                v.name
            )));
        }
    }

    let t0 = std::time::Instant::now();
    let out = session.run()?;
    eprintln!("[flint] ran in {:.2}s", t0.elapsed().as_secs_f64());

    let mut names: Vec<String> = if args.output.is_empty() {
        out.keys().cloned().collect()
    } else {
        args.output
    };
    names.sort();
    for name in names {
        let t = out
            .get(&name)
            .ok_or_else(|| flint_error::Error::Model(format!("output {name:?} not produced")))?;
        let limit = if args.full { usize::MAX } else { 8 };
        println!("{name}: {}", t.describe(limit));
    }
    Ok(())
}

fn json_to_tensor(v: &serde_json::Value) -> Result<flint_onnx::Tensor> {
    fn flatten(
        v: &serde_json::Value,
        f32s: &mut Vec<f32>,
        i64s: &mut Vec<i64>,
        bools: &mut Vec<bool>,
    ) -> Result<()> {
        match v {
            serde_json::Value::Array(a) => {
                for x in a {
                    flatten(x, f32s, i64s, bools)?;
                }
            }
            serde_json::Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    if f.fract() == 0.0 && f.abs() < 9e15 {
                        i64s.push(f as i64);
                    } else {
                        f32s.push(f as f32);
                    }
                }
            }
            serde_json::Value::Bool(b) => bools.push(*b),
            other => {
                return Err(flint_error::Error::Model(format!(
                    "unsupported input value {other:?}"
                )));
            }
        }
        Ok(())
    }
    fn shape_of(v: &serde_json::Value) -> Vec<usize> {
        let mut s = vec![];
        let mut cur = v;
        while let serde_json::Value::Array(a) = cur {
            s.push(a.len());
            cur = a.first().unwrap_or(&serde_json::Value::Null);
        }
        s
    }
    let mut f32s = vec![];
    let mut i64s = vec![];
    let mut bools = vec![];
    flatten(v, &mut f32s, &mut i64s, &mut bools)?;
    let shape = shape_of(v);
    let kinds = [
        (!f32s.is_empty()) as u8,
        (!i64s.is_empty()) as u8,
        (!bools.is_empty()) as u8,
    ];
    match kinds {
        [1, 0, 0] => Ok(flint_onnx::Tensor::f32(f32s, shape)),
        [0, 1, 0] => Ok(flint_onnx::Tensor::i64(i64s, shape)),
        [0, 0, 1] => Ok(flint_onnx::Tensor::bool(bools, shape)),
        _ => Err(flint_error::Error::Model(
            "inputs must be a homogeneous nested array of numbers or booleans".into(),
        )),
    }
}

fn chat_main(model: &Path, args: ChatArgs) -> Result<()> {
    let prompt = args
        .prompt
        .ok_or_else(|| flint_error::Error::Model("chat model requires --prompt".into()))?;
    eprintln!("[flint] initializing GPU backend...");
    let backend = Backend::new()?;
    eprintln!("[flint] adapter: {}", backend.adapter_name());

    eprintln!("[flint] loading weights from {}...", model.display());
    let load_t = std::time::Instant::now();
    let chat_model = flint_architectures::load(model, args.ctx_size, &backend)?;
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

    run_turn(
        &mut engine,
        chat.as_ref(),
        &args.system,
        &[],
        &prompt,
        args.max_tokens,
    )?;
    Ok(())
}

fn run_turn(
    engine: &mut Engine,
    chat: &dyn ChatFormat,
    system: &str,
    history: &[(String, String)],
    user: &str,
    max_tokens: usize,
) -> Result<()> {
    let text = chat.render(system, history, user);
    let mut stream = engine.stream(&text, max_tokens)?;
    for piece in stream.by_ref() {
        let piece = piece?;
        print!("{}", piece.text);
        std::io::stdout().flush().ok();
    }
    println!();
    eprintln!("{}", stream.stats().summary());
    Ok(())
}

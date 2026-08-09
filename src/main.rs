use std::io::{BufRead, Write as _};
use std::path::PathBuf;

use clap::{Args as ClapArgs, Parser, Subcommand};

use flint_architectures::chat::ChatFormat;
use flint_backend::Backend;
use flint_error::Result;
use flint_generate::{Engine, Sampler, SamplingParams};
use flint_onnx::hub;

#[derive(Parser)]
#[command(name = "flint", version, about)]
struct Args {
    #[command(subcommand)]
    cmd: Option<Command>,

    #[command(flatten)]
    chat: Option<ChatArgs>,
}

#[derive(Subcommand)]
enum Command {

    Onnx(OnnxArgs),
}

#[derive(ClapArgs)]
struct OnnxArgs {
    #[command(subcommand)]
    cmd: OnnxCmd,
}

#[derive(Subcommand)]
enum OnnxCmd {

    Download {

        repo: String,

        #[arg(long)]
        file: Option<String>,

        #[arg(long, default_value = "onnx-models")]
        out: PathBuf,
    },

    Run {

        model: PathBuf,

        #[arg(long)]
        inputs: Option<String>,

        #[arg(long)]
        inputs_file: Option<PathBuf>,

        #[arg(long)]
        output: Vec<String>,

        #[arg(long)]
        full: bool,
    },

    Info {

        model: PathBuf,
    },
}

#[derive(ClapArgs)]
struct ChatArgs {

    #[arg(long)]
    model: Option<PathBuf>,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long, default_value = "You are a helpful assistant.")]
    system: String,

    #[arg(long, default_value_t = 256)]
    max_tokens: usize,

    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    #[arg(long, default_value_t = 0.8)]
    top_p: f32,

    #[arg(long, default_value_t = 20)]
    top_k: usize,

    #[arg(long, default_value_t = 0.0)]
    min_p: f32,

    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 4096)]
    max_seq: u32,

    #[arg(long)]
    speculate: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    match args.cmd {
        Some(Command::Onnx(onnx)) => match onnx.cmd {
            OnnxCmd::Download { repo, file, out } => {
                let file = match file {
                    Some(f) => f,
                    None => hub::default_onnx_file(&repo)?,
                };
                let path = hub::download_file(&repo, &file, &out)?;
                eprintln!("[flint] model saved to {}", path.display());
                Ok(())
            }
            OnnxCmd::Run {
                model,
                inputs,
                inputs_file,
                output,
                full,
            } => onnx_run(&model, inputs, inputs_file, &output, full),
            OnnxCmd::Info { model } => onnx_info(&model),
        },
        None => chat_main(args.chat.expect("chat args present without subcommand")),
    }
}

fn onnx_run(
    model: &std::path::Path,
    inputs: Option<String>,
    inputs_file: Option<std::path::PathBuf>,
    outputs: &[String],
    full: bool,
) -> Result<()> {
    let json = match (inputs, inputs_file) {
        (Some(_), Some(_)) => {
            return Err(flint_error::Error::Model(
                "provide either --inputs or --inputs-file, not both".into(),
            ))
        }
        (Some(s), None) => s,
        (None, Some(f)) => std::fs::read_to_string(&f)
            .map_err(|e| flint_error::Error::Model(format!("cannot read {}: {e}", f.display())))?,
        (None, None) => {
            return Err(flint_error::Error::Model(
                "onnx run requires --inputs or --inputs-file".into(),
            ))
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

    let mut names: Vec<String> = if outputs.is_empty() {
        out.keys().cloned().collect()
    } else {
        outputs.to_vec()
    };
    names.sort();
    for name in names {
        let t = out.get(&name).ok_or_else(|| {
            flint_error::Error::Model(format!("output {name:?} not produced"))
        })?;
        let limit = if full { usize::MAX } else { 8 };
        println!("{name}: {}", t.describe(limit));
    }
    Ok(())
}

fn json_to_tensor(v: &serde_json::Value) -> Result<flint_onnx::Tensor> {
    fn flatten(v: &serde_json::Value, f32s: &mut Vec<f32>, i64s: &mut Vec<i64>, bools: &mut Vec<bool>) -> Result<()> {
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
                )))
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
    let kinds = [(!f32s.is_empty()) as u8, (!i64s.is_empty()) as u8, (!bools.is_empty()) as u8];
    match kinds {
        [1, 0, 0] => Ok(flint_onnx::Tensor::f32(f32s, shape)),
        [0, 1, 0] => Ok(flint_onnx::Tensor::i64(i64s, shape)),
        [0, 0, 1] => Ok(flint_onnx::Tensor::bool(bools, shape)),
        _ => Err(flint_error::Error::Model(
            "inputs must be a homogeneous nested array of numbers or booleans".into(),
        )),
    }
}

fn onnx_info(model: &std::path::Path) -> Result<()> {
    let g = flint_onnx::Graph::load(model)?;
    println!("graph: {:?}", g.name);
    println!(
        "nodes: {}  initializers: {}  inputs: {}  outputs: {}",
        g.nodes.len(),
        g.initializers.len(),
        g.inputs.len(),
        g.outputs.len()
    );
    let mut ops: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for n in &g.nodes {
        *ops.entry(n.op_type.as_str()).or_insert(0) += 1;
    }
    println!("operators:");
    for (op, count) in &ops {
        println!("  {op}: {count}");
    }
    println!("inputs:");
    for i in &g.inputs {
        println!("  {} dims={:?}", i.name, i.dims);
    }
    println!("outputs:");
    for o in &g.outputs {
        println!("  {}", o.name);
    }
    Ok(())
}

fn chat_main(args: ChatArgs) -> Result<()> {
    let model = args
        .model
        .ok_or_else(|| flint_error::Error::Model("--model is required for chat mode".into()))?;
    eprintln!("[flint] initializing GPU backend...");
    let backend = Backend::new()?;
    eprintln!("[flint] adapter: {}", backend.adapter_name());

    eprintln!("[flint] loading weights from {}...", model.display());
    let load_t = std::time::Instant::now();
    let chat_model = flint_architectures::load(&model, args.max_seq, &backend)?;
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

    match args.prompt {
        Some(prompt) => {
            run_turn(
                &mut engine,
                chat.as_ref(),
                &args.system,
                &[],
                &prompt,
                args.max_tokens,
            )?;
        }
        None => interactive(&mut engine, chat.as_ref(), &args.system, args.max_tokens)?,
    }
    if let Some(report) = engine.profile_report() {
        eprint!("{report}");
    }
    Ok(())
}

fn interactive(
    engine: &mut Engine,
    chat: &dyn ChatFormat,
    system: &str,
    max_tokens: usize,
) -> Result<()> {
    eprintln!("[flint] interactive mode. type 'exit' to quit.");
    let mut history: Vec<(String, String)> = Vec::new();
    let stdin = std::io::stdin();
    loop {
        print!("> ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let user = line.trim().to_string();
        if user.is_empty() {
            continue;
        }
        if user == "exit" {
            break;
        }
        let reply = run_turn(engine, chat, system, &history, &user, max_tokens)?;
        history.push((user, reply));
    }
    Ok(())
}

fn run_turn(
    engine: &mut Engine,
    chat: &dyn ChatFormat,
    system: &str,
    history: &[(String, String)],
    user: &str,
    max_tokens: usize,
) -> Result<String> {
    let text = chat.render(system, history, user);
    let mut stream = engine.stream(&text, max_tokens)?;
    let mut reply = String::new();
    for piece in stream.by_ref() {
        let piece = piece?;
        print!("{}", piece.text);
        std::io::stdout().flush().ok();
        reply.push_str(&piece.text);
    }
    println!();
    eprintln!("{}", stream.stats().summary());
    Ok(reply)
}

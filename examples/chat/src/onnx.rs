use std::path::Path;

use flint_error::{Error, Result};
use flint_onnx::Session;
use flint_onnx::tensor::Tensor;
use flint_tokenizer::Tokenizer;

pub fn run(model_file: &Path, dir: &Path, prompt: &str) -> Result<()> {
    let tokenizer_file = dir.join("tokenizer.json");
    if !tokenizer_file.exists() {
        return Err(Error::Model(format!(
            "onnx inference needs {} in the repo",
            tokenizer_file.display()
        )));
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_file)?;
    let ids: Vec<i64> = tokenizer
        .encode(prompt)?
        .into_iter()
        .map(i64::from)
        .collect();
    if ids.is_empty() {
        return Err(Error::Tokenizer("empty prompt".into()));
    }
    let n = ids.len();

    let mut session = Session::load(model_file)?;
    let names: Vec<String> = session.input_names().map(str::to_string).collect();
    let mut fed = 0usize;
    for name in &names {
        let t = match name.as_str() {
            "input_ids" => Tensor::i64(ids.clone(), vec![1, n]),
            "attention_mask" => Tensor::i64(vec![1; n], vec![1, n]),
            "token_type_ids" => Tensor::i64(vec![0; n], vec![1, n]),
            _ => continue,
        };
        eprintln!("[flint] input {name}: shape {:?}", t.shape);
        session.set_input(name, t)?;
        fed += 1;
    }
    if fed == 0 {
        return Err(Error::Model(format!(
            "graph {} has no input_ids/attention_mask/token_type_ids inputs",
            model_file.display()
        )));
    }

    let t0 = std::time::Instant::now();
    let out = session.run()?;
    eprintln!("[flint] ran in {:.2}s", t0.elapsed().as_secs_f64());

    let mut names: Vec<String> = out.keys().cloned().collect();
    names.sort();
    for name in names {
        let t = &out[&name];
        println!("{name}: {}", t.describe(8));
    }
    Ok(())
}

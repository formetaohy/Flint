use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use flint_error::{Error, Result};

const HUB_API: &str = "https://huggingface.co/api/models";

pub fn download_file(repo: &str, repo_file: &str, out_dir: &Path) -> Result<PathBuf> {
    let mut target = out_dir.join(repo_file);

    if let Some(name) = target.file_name() {
        target = out_dir.join(name);
    }
    if target.exists() {
        return Ok(target);
    }
    let url = format!("https://huggingface.co/{repo}/resolve/main/{repo_file}");
    eprintln!("[flint-onnx] downloading {url}");
    let resp = agent()
        .get(&url)
        .call()
        .map_err(|e| Error::Model(format!("hub request {url}: {e}")))?;
    fs::create_dir_all(out_dir)
        .map_err(|e| Error::Model(format!("cannot create {}: {e}", out_dir.display())))?;
    let mut bytes = Vec::new();
    resp.into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| Error::Model(format!("hub download {url}: {e}")))?;
    fs::write(&target, &bytes)
        .map_err(|e| Error::Model(format!("cannot write {}: {e}", target.display())))?;
    eprintln!(
        "[flint-onnx] saved {} ({} bytes)",
        target.display(),
        bytes.len()
    );
    Ok(target)
}

pub fn list_files(repo: &str) -> Result<Vec<String>> {
    let url = format!("{HUB_API}/{repo}/tree/main?recursive=true");
    let resp = agent()
        .get(&url)
        .call()
        .map_err(|e| Error::Model(format!("hub request {url}: {e}")))?;
    let mut text = String::new();
    resp.into_reader()
        .read_to_string(&mut text)
        .map_err(|e| Error::Model(format!("hub response {url}: {e}")))?;
    let entries: Vec<serde_json::Value> = serde_json::from_str(&text)
        .map_err(|e| Error::Model(format!("hub metadata {url}: {e}")))?;
    Ok(entries
        .into_iter()
        .filter_map(|v| {
            let path = v.get("path")?.as_str()?;
            Some(path.to_string())
        })
        .filter(|p| p != "onnx")
        .collect())
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(600))
        .build()
}

pub fn default_onnx_file(repo: &str) -> Result<String> {
    let files = list_files(repo)?;
    let onnx: Vec<&String> = files
        .iter()
        .filter(|f| f.ends_with(".onnx"))
        .collect();
    if onnx.is_empty() {
        return Err(Error::Model(format!(
            "repo {repo:?} has no .onnx files"
        )));
    }
    Ok(onnx
        .iter()
        .find(|f| f.as_str() == "onnx/model.onnx")
        .or_else(|| onnx.first())
        .map(|s| s.to_string())
        .unwrap())
}

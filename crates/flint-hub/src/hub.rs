use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use flint_error::{Error, Result};
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct FileEntry {
    pub path: String,
    pub size: u64,
}

pub struct Hub {
    endpoint: String,
    repo: String,
}

impl Hub {
    pub fn new(repo: &str) -> Self {
        let endpoint = std::env::var("HF_ENDPOINT")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "https://huggingface.co".to_string())
            .trim_end_matches('/')
            .to_string();
        Self {
            endpoint,
            repo: repo.to_string(),
        }
    }

    pub fn files(&self) -> Result<Vec<FileEntry>> {
        let url = format!(
            "{}/api/models/{}/tree/main?recursive=true",
            self.endpoint, self.repo
        );
        let resp = ureq::get(&url)
            .call()
            .map_err(|e| Error::Model(format!("list {url}: {e}")))?;
        let tree: Value = resp
            .into_json()
            .map_err(|e| Error::Model(format!("list {url}: {e}")))?;
        let items = tree
            .as_array()
            .ok_or_else(|| Error::Model(format!("unexpected HF API response from {url}")))?;
        Ok(items
            .iter()
            .filter(|i| i["type"] == "file")
            .map(|i| FileEntry {
                path: i["path"].as_str().unwrap_or_default().to_string(),
                size: i["size"].as_u64().unwrap_or(0),
            })
            .collect())
    }

    pub fn download(&self, path: &str, dest: &Path) -> Result<u64> {
        let url = format!("{}/{}/resolve/main/{path}", self.endpoint, self.repo);
        eprintln!("[hf] downloading {url}");
        let resp = ureq::get(&url)
            .call()
            .map_err(|e| Error::Model(format!("download {url}: {e}")))?;
        let total: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| Error::Model(format!("mkdir {}: {e}", parent.display())))?;
        }
        let part = dest.with_extension("part");
        let mut reader = resp.into_reader();
        let mut file = fs::File::create(&part)
            .map_err(|e| Error::Model(format!("create {}: {e}", part.display())))?;
        let mut buf = vec![0u8; 1 << 20];
        let mut written = 0u64;
        let mut last_pct = 0u64;
        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| Error::Model(format!("read {url}: {e}")))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])
                .map_err(|e| Error::Model(format!("write {}: {e}", part.display())))?;
            written += n as u64;
            match total {
                Some(t) if t > 0 => {
                    let pct = written * 100 / t;
                    if pct > last_pct {
                        last_pct = pct;
                        eprint!("\r[hf] {path}: {pct}% ({written} / {t} bytes)");
                        std::io::stderr().flush().ok();
                    }
                }
                _ if written.is_multiple_of(64 << 20) => {
                    eprint!("\r[hf] {path}: {written} bytes");
                    std::io::stderr().flush().ok();
                }
                _ => {}
            }
        }
        file.flush()
            .map_err(|e| Error::Model(format!("flush {}: {e}", part.display())))?;
        drop(file);
        if last_pct > 0 || total.is_none() {
            eprintln!();
        }
        fs::rename(&part, dest).map_err(|e| {
            Error::Model(format!(
                "rename {} -> {}: {e}",
                part.display(),
                dest.display()
            ))
        })?;
        Ok(written)
    }
}

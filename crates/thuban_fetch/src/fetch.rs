use std::fs;
use std::path::{Path, PathBuf};

use thuban_error::{Error, Result};

use crate::repo::{FileEntry, Repo};

pub fn fetch(repo: &Repo, dir: &Path) -> Result<()> {
    fs::create_dir_all(dir).map_err(|e| Error::Model(format!("mkdir {}: {e}", dir.display())))?;
    if let Some(local) = find_local_gguf(dir) {
        eprintln!("[hf] using local {}", local.display());
        return Ok(());
    }
    let files = repo.files()?;
    for entry in select_gguf(&files)? {
        let dest = dir.join(&entry.path);
        let complete = dest
            .metadata()
            .map(|m| m.is_file() && m.len() == entry.size)
            .unwrap_or(false);
        if complete {
            eprintln!("[hf] {} present, skipping", entry.path);
        } else {
            let written = repo.download(&entry.path, &dest)?;
            if written != entry.size {
                return Err(Error::Model(format!(
                    "{}: downloaded {written} bytes, expected {}",
                    entry.path, entry.size
                )));
            }
        }
    }
    Ok(())
}

fn find_local_gguf(dir: &Path) -> Option<PathBuf> {
    fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "gguf"))
        .max_by_key(|p| p.metadata().map(|m| m.len()).unwrap_or(0))
}

fn select_gguf(files: &[FileEntry]) -> Result<Vec<FileEntry>> {
    let mut matches: Vec<FileEntry> = files
        .iter()
        .filter(|f| f.path.ends_with(".gguf"))
        .cloned()
        .collect();
    if matches.is_empty() {
        return Err(Error::Model("repo has no .gguf file".into()));
    }
    matches.sort_by_key(|f| std::cmp::Reverse(f.size));
    if matches.len() > 1 {
        eprintln!(
            "[hf] {} .gguf files, using largest: {} ({:.1} MiB)",
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

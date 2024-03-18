use std::fs;
use std::path::{Path, PathBuf};

pub fn find_files_with_extension(dir: &str, ext: &str) -> std::io::Result<Vec<PathBuf>> {
    let mut files_with_extension = Vec::new();
    visit_dirs(Path::new(dir), ext, &mut files_with_extension)?;
    Ok(files_with_extension)
}

fn visit_dirs(dir: &Path, ext: &str, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, ext, files)?;
            } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
                files.push(path);
            }
        }
    }
    Ok(())
}

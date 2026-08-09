use std::process::Command;

use saturn_compiler::{Source, compile};

fn main() {
    let source = Source::new("scl/blk.scl", include_str!("scl/blk.scl"));
    let kernel = compile(&source).expect("compile");
    let spirv = saturn_shader::to_spirv(&kernel).expect("spirv");
    std::fs::write("rt.spv", &spirv).expect("write");
    let out = Command::new(r"C:\VulkanSDK\1.4.328.1\Bin\spirv-val.exe")
        .arg("--target-env")
        .arg("vulkan1.3")
        .arg("rt.spv")
        .output()
        .expect("val");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if text.trim().is_empty() {
        println!("VALID");
    } else {
        println!("{text}");
    }
}

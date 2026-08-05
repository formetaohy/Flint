fn main() {
    println!("cargo:rerun-if-changed=proto/onnx.proto");
    let fds = protox::compile(["proto/onnx.proto"], ["proto"]).expect("parse onnx.proto");
    prost_build::compile_fds(fds).expect("codegen onnx.proto");
}

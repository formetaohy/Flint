mod attn;
mod bandwidth;
mod caps;
mod cpu;
mod gemm;
mod gemv;
mod paged;

pub fn run(name: &str) -> flint_error::Result<()> {
    use flint_error::Error;
    match name {
        "caps" => caps::caps_probe(),
        "bandwidth" => bandwidth::bandwidth_probe(),
        "gemv" => gemv::gemv_probe(),
        "cpu" => cpu::cpu_probe(),
        "gemm" => gemm::gemm_probe(),
        "attn" => attn::attn_probe(),
        "paged" => paged::paged_probe(),
        other => Err(Error::Model(format!("unknown probe {other:?}"))),
    }
}

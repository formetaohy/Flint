mod attn;
mod bandwidth;
mod caps;
mod cpu;
mod gemm;
mod gemv;
mod dispatch;
mod paged;

pub fn run(name: &str) -> thuban_error::Result<()> {
    use thuban_error::Error;
    match name {
        "caps" => caps::caps_probe(),
        "bandwidth" => bandwidth::bandwidth_probe(),
        "gemv" => gemv::gemv_probe(),
        "cpu" => cpu::cpu_probe(),
        "gemm" => gemm::gemm_probe(),
        "attn" => attn::attn_probe(),
        "dispatch" => dispatch::dispatch_probe(),
        "paged" => paged::paged_probe(),
        other => Err(Error::Model(format!("unknown probe {other:?}"))),
    }
}

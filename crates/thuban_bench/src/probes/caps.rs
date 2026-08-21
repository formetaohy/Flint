use thuban_backend::Backend;
use thuban_error::Result;

pub(super) fn caps_probe() -> Result<()> {
    let backend = Backend::new()?;
    eprintln!("[probe] adapter: {}", backend.adapter_name());
    let device = backend.device();
    eprintln!(
        "[probe] subgroup size: {}-{}",
        device.subgroup_min_size(),
        device.subgroup_max_size()
    );
    let props = device.cooperative_matrix_properties();
    eprintln!("[probe] cooperative matrix configs: {}", props.len());
    for p in props {
        eprintln!(
            "[probe]   coop {}x{}x{} a/b={:?} c/r={:?} saturating={}",
            p.m_size, p.n_size, p.k_size, p.ab_type, p.cr_type, p.saturating_accumulation
        );
    }
    eprintln!("[probe] gemm coop variant: {:?}", device.coop_gemm());
    Ok(())
}

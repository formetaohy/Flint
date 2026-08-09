#[derive(Debug, Clone, Copy)]
pub struct BufferSpec {
    pub size: u64,
    pub host_visible: bool,
}

#[derive(Debug, Clone)]
pub struct KernelSpec {
    pub name: String,
    pub source: &'static str,
}

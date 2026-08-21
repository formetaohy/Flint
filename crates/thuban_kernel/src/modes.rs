#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Act {
    Silu = 0,
    GeluTanh = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum NormMode {
    Offset = 0,
    Gated = 1,
    Direct = 2,
    Layer = 3,
}

use crate::kernel::Scalar;

#[derive(Debug, Clone, Copy)]
pub struct BufferSpec {
    pub size: u64,
    pub host_visible: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixRole {
    A,
    B,
    Acc,
}

impl MatrixRole {
    pub fn encode(self) -> u32 {
        match self {
            MatrixRole::A => 0,
            MatrixRole::B => 1,
            MatrixRole::Acc => 2,
        }
    }

    pub fn decode(code: u32) -> Option<MatrixRole> {
        match code {
            0 => Some(MatrixRole::A),
            1 => Some(MatrixRole::B),
            2 => Some(MatrixRole::Acc),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PrecompiledScalar<'a> {
    pub name: &'a str,
    pub offset: u32,
    pub ty: Scalar,
}

#[derive(Debug, Clone, Copy)]
pub struct PrecompiledKernel<'a> {
    pub name: &'a str,
    pub workgroup_size: [u32; 3],
    pub bindings: &'a [u32],
    pub spirv: &'a [u8],
    pub msl: &'a str,
    pub scalars: &'a [PrecompiledScalar<'a>],
    pub coop_triples: &'a [(Scalar, Scalar, Scalar)],
    pub coop_roles: &'a [(Scalar, u32)],
}

#[derive(Debug, Clone)]
pub struct KernelSpec<'a> {
    pub name: String,
    pub source: &'a str,
    pub specs: &'a [(&'a str, f64)],
    pub precompiled: Option<&'a PrecompiledKernel<'a>>,
}

impl<'a> KernelSpec<'a> {
    pub fn new(name: impl Into<String>, source: &'a str) -> Self {
        Self {
            name: name.into(),
            source,
            specs: &[],
            precompiled: None,
        }
    }

    pub fn with_specs(mut self, specs: &'a [(&'a str, f64)]) -> Self {
        self.specs = specs;
        self
    }

    pub fn precompiled(name: impl Into<String>, pc: &'a PrecompiledKernel<'a>) -> Self {
        Self {
            name: name.into(),
            source: pc.name,
            specs: &[],
            precompiled: Some(pc),
        }
    }
}

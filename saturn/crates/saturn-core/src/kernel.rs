use std::any::Any;

use crate::num::f32_to_f16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scalar {
    F32,
    F16,
    Bf16,
    I32,
    U32,
    I8,
    U8,
    Bool,
}

impl Scalar {
    pub fn width(&self) -> u32 {
        match self {
            Scalar::F32 | Scalar::I32 | Scalar::U32 => 4,
            Scalar::F16 | Scalar::Bf16 => 2,
            Scalar::I8 | Scalar::U8 | Scalar::Bool => 1,
        }
    }

    pub fn encode(&self, value: f64) -> [u8; 4] {
        let bits = match self {
            Scalar::F32 => (value as f32).to_bits(),
            Scalar::F16 => f32_to_f16(value as f32) as u32,
            Scalar::Bf16 => (value as f32).to_bits() >> 16,
            Scalar::I32 => (value as i32) as u32,
            Scalar::U32 => value as u32,
            Scalar::I8 => (value as i8) as u8 as u32,
            Scalar::U8 => value as u8 as u32,
            Scalar::Bool => (value != 0.0) as u8 as u32,
        };
        bits.to_le_bytes()
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Scalar::F32 | Scalar::F16)
    }

    pub fn is_int(&self) -> bool {
        matches!(self, Scalar::I32 | Scalar::U32 | Scalar::I8 | Scalar::U8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(scalar: Scalar, value: f64) -> u32 {
        u32::from_le_bytes(scalar.encode(value))
    }

    #[test]
    fn f16_encodes_through_num_conversion() {
        assert_eq!(bits(Scalar::F16, 1.0), 0x3C00);
        assert_eq!(bits(Scalar::F16, 2.0), 0x4000);
        assert_eq!(bits(Scalar::F16, 65504.0), 0x7BFF);
    }

    #[test]
    fn bf16_truncates_f32_bits() {
        assert_eq!(bits(Scalar::Bf16, 1.0), 0x3F80);
        assert_eq!(bits(Scalar::Bf16, -2.0), 0xC000);
        assert_eq!(bits(Scalar::Bf16, 1.5), 0x3FC0);
    }

    #[test]
    fn integer_scalars_encode_little_endian() {
        assert_eq!(bits(Scalar::F32, 1.5), 0x3FC0_0000);
        assert_eq!(bits(Scalar::I32, -2.0), 0xFFFF_FFFE);
        assert_eq!(bits(Scalar::U32, 0xDEAD_BEEFu32 as f64), 0xDEAD_BEEF);
        assert_eq!(bits(Scalar::I8, -1.0), 0xFF);
        assert_eq!(bits(Scalar::U8, 200.0), 200);
        assert_eq!(bits(Scalar::Bool, 1.0), 1);
        assert_eq!(bits(Scalar::Bool, 0.0), 0);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarField {
    pub name: String,
    pub offset: u32,
    pub ty: Scalar,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScalarLayout {
    pub size: u32,
    pub fields: Vec<ScalarField>,
}

pub trait Kernel: Any {
    fn name(&self) -> &str;
    fn workgroup_size(&self) -> [u32; 3];
    fn scalar_layout(&self) -> Option<&ScalarLayout>;
    fn as_any(&self) -> &dyn Any;
}

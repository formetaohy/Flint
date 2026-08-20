#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scalar {
    F32,
    U32,
}

impl Scalar {
    pub fn width(&self) -> u32 {
        match self {
            Scalar::F32 | Scalar::U32 => 4,
        }
    }

    pub fn encode(&self, value: f64) -> [u8; 4] {
        let bits = match self {
            Scalar::F32 => (value as f32).to_bits(),
            Scalar::U32 => value as u32,
        };
        bits.to_le_bytes()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(scalar: Scalar, value: f64) -> u32 {
        u32::from_le_bytes(scalar.encode(value))
    }

    #[test]
    fn f32_encodes_through_bit_cast() {
        assert_eq!(bits(Scalar::F32, 1.5), 0x3FC0_0000);
        assert_eq!(bits(Scalar::F32, -2.0), 0xC000_0000);
    }

    #[test]
    fn u32_encodes_little_endian() {
        assert_eq!(bits(Scalar::U32, 0xDEAD_BEEFu32 as f64), 0xDEAD_BEEF);
    }
}

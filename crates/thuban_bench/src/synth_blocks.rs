use thuban_tensor::Quant;

pub fn synth_blocks(quant: Quant, n: u32, k: u32) -> Vec<u8> {
    let bl = quant.block_len() as u32;
    let bb = quant.block_bytes();
    let blocks = (n as usize) * (k as usize / bl as usize);
    let mut raw = vec![0u8; blocks * bb];
    for b in 0..blocks {
        let off = b * bb;
        match quant {
            Quant::Q8_0 => {
                raw[off..off + 2].copy_from_slice(&0x3800u16.to_le_bytes());
                for (i, q) in raw[off + 2..off + bb].iter_mut().enumerate() {
                    *q = ((i * 7 + b) % 251) as u8;
                }
            }
            Quant::Q4K => {
                raw[off..off + 2].copy_from_slice(&0x3800u16.to_le_bytes());
                raw[off + 2..off + 4].copy_from_slice(&0x2c00u16.to_le_bytes());
                for (i, q) in raw[off + 4..off + bb].iter_mut().enumerate() {
                    *q = ((i * 13 + b) % 251) as u8;
                }
            }
            _ => {
                for i in 0..bb {
                    raw[off + i] = ((i * 13 + b) % 251) as u8;
                }
            }
        }
    }
    quant.pad_blocks(&raw, (n as usize) * (k as usize)).expect("synth blocks")
}

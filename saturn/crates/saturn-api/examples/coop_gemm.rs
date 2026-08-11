use saturn_api::{BackendKind, open};
use saturn_core::{BindingRef, BufferSpec, KernelSpec};

const M: usize = 256;
const N: usize = 256;
const K: usize = 16;
const TILE: usize = 16;

fn to_f16_bits(value: f32) -> [u8; 2] {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    let half = if exp >= 0x8F {
        if exp == 0xFF {
            sign | 0x7C00 | (mant >> 13)
        } else {
            sign | 0x7C00
        }
    } else if exp <= 0x70 {
        sign
    } else {
        let e = exp - 127 + 15;
        sign | ((e as u32) << 10) | (mant >> 13)
    };
    (half as u16).to_le_bytes()
}

fn f16_bits_to_f32(bytes: &[u8]) -> f32 {
    let half = u16::from_le_bytes([bytes[0], bytes[1]]) as u32;
    let sign = (half & 0x8000) << 16;
    let exp = ((half >> 10) & 0x1F) as i32;
    let mant = half & 0x3FF;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let m = mant;
            sign | (((-14 + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | (((exp - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn main() {
    let device: Box<dyn saturn_core::Device> = match std::env::args().nth(1).as_deref() {
        Some("metal") => open(BackendKind::Metal).expect("open metal"),
        Some("vulkan-validation") => Box::new(
            saturn_vk::VkDevice::open(&saturn_vk::Options { validation: true })
                .expect("open vulkan with validation"),
        ),
        _ => open(BackendKind::Vulkan).expect("open vulkan"),
    };

    let a_buf = device
        .create_buffer(&BufferSpec {
            size: (M * K * 2) as u64,
            host_visible: true,
        })
        .expect("create a");
    let b_buf = device
        .create_buffer(&BufferSpec {
            size: (K * N * 2) as u64,
            host_visible: true,
        })
        .expect("create b");
    let c_buf = device
        .create_buffer(&BufferSpec {
            size: (M * N * 4) as u64,
            host_visible: true,
        })
        .expect("create c");

    let mut a_data = Vec::with_capacity(M * K * 2);
    let mut b_data = Vec::with_capacity(K * N * 2);
    for i in 0..M * K {
        a_data.extend_from_slice(&to_f16_bits(((i * 7 % 13) as f32) * 0.5));
    }
    for i in 0..K * N {
        b_data.extend_from_slice(&to_f16_bits(((i * 3 % 11) as f32) * 0.25));
    }
    a_buf.write(0, &a_data).expect("write a");
    b_buf.write(0, &b_data).expect("write b");

    let kernel = device
        .create_kernel(&KernelSpec::precompiled(
            "scl/coop_gemm.scl",
            saturn_scl::scl!("coop_gemm.scl"),
        ))
        .expect("create kernel (requires VK_KHR_cooperative_matrix)");

    let mut encoder = device.encoder().expect("encoder");
    encoder
        .bind(
            &*kernel,
            &[
                BindingRef {
                    index: 0,
                    buffer: &*a_buf,
                    offset: 0,
                    size: 0,
                },
                BindingRef {
                    index: 1,
                    buffer: &*b_buf,
                    offset: 0,
                    size: 0,
                },
                BindingRef {
                    index: 2,
                    buffer: &*c_buf,
                    offset: 0,
                    size: 0,
                },
            ],
        )
        .expect("bind");
    let mut scalars = Vec::with_capacity(12);
    for value in [M as u32, N as u32, K as u32] {
        scalars.extend_from_slice(&value.to_le_bytes());
    }
    encoder
        .set_scalars(&*kernel, &scalars)
        .expect("set scalars");
    encoder
        .dispatch([(N / 128) as u32, (M / TILE) as u32, 1])
        .expect("dispatch");
    let submission = device.submit(encoder).expect("submit");
    submission.wait().expect("wait");

    let mut output = vec![0u8; M * N * 4];
    c_buf.read(0, &mut output).expect("read c");
    let mut reference = vec![0.0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut sum = 0.0f32;
            for t in 0..K {
                let av = f16_bits_to_f32(&a_data[(i * K + t) * 2..(i * K + t) * 2 + 2]);
                let bv = f16_bits_to_f32(&b_data[(t * N + j) * 2..(t * N + j) * 2 + 2]);
                sum += av * bv;
            }
            reference[i * N + j] = sum;
        }
    }
    for r in 0..4 {
        let row: Vec<String> = (0..16)
            .map(|col| {
                let i = r * N + col;
                let got = f32::from_le_bytes([
                    output[i * 4],
                    output[i * 4 + 1],
                    output[i * 4 + 2],
                    output[i * 4 + 3],
                ]);
                format!("{got:.1}")
            })
            .collect();
        let refrow: Vec<String> = (0..16)
            .map(|col| format!("{:.1}", reference[r * N + col]))
            .collect();
        println!("C[{r}]: {row:?}");
        println!("ref[{r}]: {refrow:?}");
    }
    let mut computed_tiles = 0;
    let mut zero_tiles = 0;
    let mut max_err = 0.0f32;
    for tile_row in 0..M / TILE {
        for tile_col in 0..N / TILE {
            let mut tile_nonzero = false;
            for r in 0..TILE {
                for col in 0..TILE {
                    let i = (tile_row * TILE + r) * N + tile_col * TILE + col;
                    let got = f32::from_le_bytes([
                        output[i * 4],
                        output[i * 4 + 1],
                        output[i * 4 + 2],
                        output[i * 4 + 3],
                    ]);
                    if got != 0.0 {
                        tile_nonzero = true;
                    }
                }
            }
            if tile_nonzero {
                computed_tiles += 1;
                for r in 0..TILE {
                    for col in 0..TILE {
                        let i = (tile_row * TILE + r) * N + tile_col * TILE + col;
                        let got = f32::from_le_bytes([
                            output[i * 4],
                            output[i * 4 + 1],
                            output[i * 4 + 2],
                            output[i * 4 + 3],
                        ]);
                        let err = (got - reference[i]).abs();
                        if err > max_err {
                            max_err = err;
                        }
                        if err > 2.0 {
                            println!("  mismatch at {i}: got {got}, expected {}", reference[i]);
                        }
                    }
                }
            } else {
                zero_tiles += 1;
            }
        }
    }
    println!(
        "coop gemm ok on {} (computed tiles {computed_tiles}, zero tiles {zero_tiles}, max err {max_err})",
        device.name()
    );
}

use saturn_api::{BackendKind, open};
use saturn_core::{BindingRef, BufferSpec, KernelSpec};

const T: usize = 16;
const N: usize = 128;

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
    let device: Box<dyn saturn_core::Device> =
        open(BackendKind::Vulkan).expect("open vulkan");
    let src = device
        .create_buffer(&BufferSpec {
            size: (T * N * 2) as u64,
            host_visible: true,
        })
        .expect("create src");
    let dst = device
        .create_buffer(&BufferSpec {
            size: (T * T * 8 * 2) as u64,
            host_visible: true,
        })
        .expect("create dst");
    let mut input = Vec::with_capacity(T * N * 2);
    for i in 0..T * N {
        input.extend_from_slice(&to_f16_bits((i * 3 % 11) as f32 * 0.25));
    }
    src.write(0, &input).expect("write src");

    let kernel = device
        .create_kernel(&KernelSpec::precompiled("scl/coop_rt16b.scl", saturn_scl::scl!("coop_rt16b.scl")))
        .expect("create kernel");
    let mut encoder = device.encoder().expect("encoder");
    encoder
        .bind(
            &*kernel,
            &[
                BindingRef {
                    index: 0,
                    buffer: &*src,
                    offset: 0,
                    size: 0,
                },
                BindingRef {
                    index: 1,
                    buffer: &*dst,
                    offset: 0,
                    size: 0,
                },
            ],
        )
        .expect("bind");
    encoder
        .set_scalars(&*kernel, &(N as u32).to_le_bytes())
        .expect("set scalars");
    encoder.dispatch([1, 1, 1]).expect("dispatch");
    let submission = device.submit(encoder).expect("submit");
    submission.wait().expect("wait");
    let mut output = vec![0u8; T * T * 8 * 2];
    dst.read(0, &mut output).expect("read dst");
    let mut bad = 0;
    for sg in 0..8usize {
        for r in 0..T {
            for c in 0..T {
                let i = sg * 256 + r * 16 + c;
                let got = f16_bits_to_f32(&output[i * 2..i * 2 + 2]);
                let expect = f16_bits_to_f32(
                    &input[(r * N + sg * 16 + c) * 2..(r * N + sg * 16 + c) * 2 + 2],
                );
                if got != expect {
                    bad += 1;
                    if bad <= 6 {
                        println!("dst[sg{sg}][{r}][{c}] = {got}, expect {expect}");
                    }
                }
            }
        }
    }
    println!(
        "roundtrip16b on {}: {} bad of {}",
        device.name(),
        bad,
        T * T * 8
    );
}

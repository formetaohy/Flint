use saturn_api::{BackendKind, open};
use saturn_core::{BindingRef, BufferSpec, KernelSpec};

const T: usize = 16;

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

fn main() {
    let device: Box<dyn saturn_core::Device> =
        open(BackendKind::Vulkan).expect("open vulkan");
    let src = device
        .create_buffer(&BufferSpec {
            size: (T * T * 2) as u64,
            host_visible: true,
        })
        .expect("create src");
    let dst = device
        .create_buffer(&BufferSpec {
            size: (T * T * 4) as u64,
            host_visible: true,
        })
        .expect("create dst");
    let mut input = vec![0u8; T * T * 2];
    for i in 0..T {
        input[i * 2 + i * T * 2..i * 2 + i * T * 2 + 2].copy_from_slice(&to_f16_bits(1.0));
    }
    src.write(0, &input).expect("write src");

    let kernel = device
        .create_kernel(&KernelSpec {
            name: "sat/coop_id16.sat".into(),
            source: saturn_sat::sat!("coop_id16.sat"),
        })
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
    encoder.dispatch([1, 1, 1]).expect("dispatch");
    let submission = device.submit(encoder).expect("submit");
    submission.wait().expect("wait");
    let mut output = vec![0u8; T * T * 4];
    dst.read(0, &mut output).expect("read dst");
    let mut bad = 0;
    for r in 0..T {
        for c in 0..T {
            let i = r * T + c;
            let got = f32::from_le_bytes([
                output[i * 4],
                output[i * 4 + 1],
                output[i * 4 + 2],
                output[i * 4 + 3],
            ]);
            let expect = if r == c { 1.0 } else { 0.0 };
            if got != expect {
                bad += 1;
                if bad <= 6 {
                    println!("out[{r}][{c}] = {got}, expect {expect}");
                }
            }
        }
    }
    println!("identity16 on {}: {} bad of {}", device.name(), bad, T * T);
}

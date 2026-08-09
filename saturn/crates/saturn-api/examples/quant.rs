use saturn_api::{BackendKind, open};
use saturn_core::{BindingRef, BufferSpec, KernelSpec};

const N: usize = 1024;

fn main() {
    let device: Box<dyn saturn_core::Device> = match std::env::args().nth(1).as_deref() {
        Some("metal") => open(BackendKind::Metal).expect("open metal"),
        Some("vulkan-validation") => Box::new(
            saturn_vk::VkDevice::open(&saturn_vk::Options { validation: true })
                .expect("open vulkan with validation"),
        ),
        _ => open(BackendKind::Vulkan).expect("open vulkan"),
    };
    let src = device
        .create_buffer(&BufferSpec {
            size: (N * 4) as u64,
            host_visible: true,
        })
        .expect("create src");
    let w = device
        .create_buffer(&BufferSpec {
            size: (N * 2) as u64,
            host_visible: true,
        })
        .expect("create w");
    let q = device
        .create_buffer(&BufferSpec {
            size: N as u64,
            host_visible: true,
        })
        .expect("create q");
    let mut input = Vec::with_capacity(N * 4);
    for i in 0..N {
        let v = ((i % 100) as f32 - 50.0) * 0.7;
        input.extend_from_slice(&v.to_le_bytes());
    }
    src.write(0, &input).expect("write src");
    let kernel = device
        .create_kernel(&KernelSpec::precompiled("sat/quant.sat", saturn_sat::sat!("quant.sat")))
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
                    buffer: &*w,
                    offset: 0,
                    size: 0,
                },
                BindingRef {
                    index: 2,
                    buffer: &*q,
                    offset: 0,
                    size: 0,
                },
            ],
        )
        .expect("bind");
    encoder
        .set_scalars(&*kernel, &1.5f32.to_le_bytes())
        .expect("set scalars");
    encoder.dispatch([(N / 64) as u32, 1, 1]).expect("dispatch");
    let submission = device.submit(encoder).expect("submit");
    submission.wait().expect("wait");

    let mut w_out = vec![0u8; N * 2];
    let mut q_out = vec![0u8; N];
    w.read(0, &mut w_out).expect("read w");
    q.read(0, &mut q_out).expect("read q");

    let mut max_bf16_err = 0.0f32;
    for i in 0..N {
        let v = f32::from_le_bytes([
            input[i * 4],
            input[i * 4 + 1],
            input[i * 4 + 2],
            input[i * 4 + 3],
        ]);
        let bits = u16::from_le_bytes([w_out[i * 2], w_out[i * 2 + 1]]) as u32;
        let r = f32::from_bits(bits << 16);
        let err = (r - v).abs();
        if err > max_bf16_err {
            max_bf16_err = err;
        }
        let scaled = (r * 1.5).clamp(0.0, 255.0) as u8;
        let got_q = q_out[i];
        assert!(
            (got_q as i32 - scaled as i32).abs() <= 1,
            "quant {i}: got {got_q}, expected {scaled}"
        );
    }
    println!(
        "quant ok on {} (max bf16 err {max_bf16_err})",
        device.name()
    );
}

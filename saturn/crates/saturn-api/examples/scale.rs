use std::env;

use saturn_api::{BackendKind, open};
use saturn_core::{BindingRef, BufferSpec, KernelSpec};

const N: usize = 1024;

fn main() {
    let device: Box<dyn saturn_core::Device> = match env::args().nth(1).as_deref() {
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
        .expect("create src buffer");
    let dst = device
        .create_buffer(&BufferSpec {
            size: (N * 4) as u64,
            host_visible: true,
        })
        .expect("create dst buffer");
    let kernel = device
        .create_kernel(&KernelSpec::precompiled("scl/scale.scl", saturn_scl::scl!("scale.scl")))
        .expect("create kernel");
    let mut input = Vec::with_capacity(N * 4);
    for i in 0..N {
        input.extend_from_slice(&(i as f32).to_le_bytes());
    }
    src.write(0, &input).expect("write src");
    let mut encoder = device.encoder().expect("create encoder");
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
    encoder.dispatch([(N / 64) as u32, 1, 1]).expect("dispatch");
    let submission = device.submit(encoder).expect("submit");
    submission.wait().expect("wait");
    let mut output = vec![0u8; N * 4];
    dst.read(0, &mut output).expect("read dst");
    for i in 0..N {
        let v = f32::from_le_bytes([
            output[i * 4],
            output[i * 4 + 1],
            output[i * 4 + 2],
            output[i * 4 + 3],
        ]);
        assert_eq!(v, (i as f32) * 2.0, "index {i}");
    }
    println!("scale ok on {}", device.name());
}

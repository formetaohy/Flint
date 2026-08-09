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
    let a = device
        .create_buffer(&BufferSpec {
            size: (N * 4) as u64,
            host_visible: true,
        })
        .expect("create a");
    let out = device
        .create_buffer(&BufferSpec {
            size: 8,
            host_visible: true,
        })
        .expect("create out");
    let mut input = Vec::with_capacity(N * 4);
    for i in 0..N {
        input.extend_from_slice(&((i % 7 + 1) as u32).to_le_bytes());
    }
    a.write(0, &input).expect("write a");
    out.write(0, &0u32.to_le_bytes()).expect("init out");
    let kernel = device
        .create_kernel(&KernelSpec::precompiled("scl/atomic.scl", saturn_scl::scl!("atomic.scl")))
        .expect("create kernel");
    let mut encoder = device.encoder().expect("encoder");
    encoder
        .bind(
            &*kernel,
            &[
                BindingRef {
                    index: 0,
                    buffer: &*a,
                    offset: 0,
                    size: 0,
                },
                BindingRef {
                    index: 1,
                    buffer: &*out,
                    offset: 0,
                    size: 0,
                },
            ],
        )
        .expect("bind");
    encoder.dispatch([(N / 64) as u32, 1, 1]).expect("dispatch");
    let submission = device.submit(encoder).expect("submit");
    submission.wait().expect("wait");
    let expect: u32 = (0..N).map(|i| (i % 7 + 1) as u32).sum();
    let mut output = vec![0u8; 8];
    out.read(0, &mut output).expect("read out");
    let got = u32::from_le_bytes([output[0], output[1], output[2], output[3]]);
    let snap = u32::from_le_bytes([output[4], output[5], output[6], output[7]]);
    assert_eq!(got, expect, "atomic sum mismatch");
    assert_eq!(snap, expect, "snapshot mismatch");
    println!("atomic ok on {} (sum {got})", device.name());
}

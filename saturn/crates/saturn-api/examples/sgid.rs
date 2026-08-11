use saturn_api::{BackendKind, open};
use saturn_core::{BindingRef, BufferSpec, KernelSpec};

fn main() {
    let device: Box<dyn saturn_core::Device> = open(BackendKind::Vulkan).expect("open vulkan");
    let buf = device
        .create_buffer(&BufferSpec {
            size: 128 * 4,
            host_visible: true,
        })
        .expect("create");
    let kernel = device
        .create_kernel(&KernelSpec::precompiled(
            "scl/sgid.scl",
            saturn_scl::scl!("sgid.scl"),
        ))
        .expect("create kernel");
    let mut encoder = device.encoder().expect("encoder");
    encoder
        .bind(
            &*kernel,
            &[BindingRef {
                index: 0,
                buffer: &*buf,
                offset: 0,
                size: 0,
            }],
        )
        .expect("bind");
    encoder.dispatch([2, 4, 1]).expect("dispatch");
    let submission = device.submit(encoder).expect("submit");
    submission.wait().expect("wait");
    let mut out = vec![0u8; 128 * 4];
    buf.read(0, &mut out).expect("read");
    let mut bad = 0;
    for i in 0..128 {
        let v = u32::from_le_bytes([out[i * 4], out[i * 4 + 1], out[i * 4 + 2], out[i * 4 + 3]]);
        let gx = i % 8;
        let gy = i / 8;
        let expect = ((gy / 4) * 16 + gx / 4) as u32;
        if v != expect {
            bad += 1;
            if bad <= 4 {
                println!("gid({gx},{gy}) -> {v}, expect {expect}");
            }
        }
    }
    println!("block ids: {bad} bad");
}

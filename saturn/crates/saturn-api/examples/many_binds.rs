use saturn_api::{BackendKind, open};
use saturn_core::{BindingRef, BufferSpec, KernelSpec};

fn main() {
    let device: Box<dyn saturn_core::Device> =
        open(BackendKind::Vulkan).expect("open vulkan");
    let count = 256usize;
    let src = device
        .create_buffer(&BufferSpec {
            size: (count * 4) as u64,
            host_visible: true,
        })
        .expect("create src");
    src.write(0, &vec![1.0f32.to_le_bytes(); count].concat())
        .expect("fill src");
    let dst = device
        .create_buffer(&BufferSpec {
            size: (count * 4) as u64,
            host_visible: true,
        })
        .expect("create dst");
    let kernel = device
        .create_kernel(&KernelSpec::precompiled("scl/scale.scl", saturn_scl::scl!("scale.scl")))
        .expect("create kernel");
    let mut encoder = device.encoder().expect("encoder");
    let bindings = [
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
    ];
    for i in 0..1000u32 {
        encoder.bind(&*kernel, &bindings).expect("bind");
        encoder.dispatch([4, 1, 1]).expect("dispatch");
        if i % 10 == 9 {
            encoder.barrier().expect("barrier");
        }
    }
    let submission = device.submit(encoder).expect("submit");
    submission.wait().expect("wait");
    let mut out = vec![0u8; count * 4];
    dst.read(0, &mut out).expect("read");
    let mut bad = 0;
    for i in 0..count {
        let v = f32::from_le_bytes([out[i * 4], out[i * 4 + 1], out[i * 4 + 2], out[i * 4 + 3]]);
        if v != 2.0 {
            bad += 1;
        }
    }
    println!(
        "many_binds ok on {} ({} binds, {} pools, {bad} nonzero)",
        device.name(),
        1000,
        1000 / 256 + 1
    );
    assert_eq!(bad, 0);
}

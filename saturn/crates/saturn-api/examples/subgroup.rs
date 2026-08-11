use saturn_api::{BackendKind, open};
use saturn_core::{BindingRef, BufferSpec, KernelSpec};

const N: usize = 4096;

fn main() {
    let device: Box<dyn saturn_core::Device> = match std::env::args().nth(1).as_deref() {
        Some("metal") => open(BackendKind::Metal).expect("open metal"),
        Some("vulkan-validation") => Box::new(
            saturn_vk::VkDevice::open(&saturn_vk::Options { validation: true })
                .expect("open vulkan with validation"),
        ),
        _ => open(BackendKind::Vulkan).expect("open vulkan"),
    };

    let size = (N * 4) as u64;
    let a_buf = device
        .create_buffer(&BufferSpec {
            size,
            host_visible: true,
        })
        .expect("create a");
    let b_buf = device
        .create_buffer(&BufferSpec {
            size,
            host_visible: true,
        })
        .expect("create b");
    let c_buf = device
        .create_buffer(&BufferSpec {
            size,
            host_visible: true,
        })
        .expect("create c");
    let d_buf = device
        .create_buffer(&BufferSpec {
            size,
            host_visible: true,
        })
        .expect("create d");

    let mut input = Vec::with_capacity(N * 4);
    for i in 0..N {
        input.extend_from_slice(&((i % 100) as f32).to_le_bytes());
    }
    a_buf.write(0, &input).expect("write a");

    let kernel = device
        .create_kernel(&KernelSpec::precompiled(
            "scl/subgroup.scl",
            saturn_scl::scl!("subgroup.scl"),
        ))
        .expect("create kernel");

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
                BindingRef {
                    index: 3,
                    buffer: &*d_buf,
                    offset: 0,
                    size: 0,
                },
            ],
        )
        .expect("bind");
    encoder.dispatch([(N / 64) as u32, 1, 1]).expect("dispatch");
    let submission = device.submit(encoder).expect("submit");
    submission.wait().expect("wait");

    let mut b_out = vec![0u8; N * 4];
    let mut c_out = vec![0u8; N * 4];
    let mut d_out = vec![0u8; N * 4];
    b_buf.read(0, &mut b_out).expect("read b");
    c_buf.read(0, &mut c_out).expect("read c");
    d_buf.read(0, &mut d_out).expect("read d");

    let a: Vec<f32> = input
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let read = |buf: &[u8], i: usize| {
        f32::from_le_bytes([buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]])
    };

    let mut subgroups = 0;
    let mut start = 0;
    for i in 0..N {
        let same = read(&b_out, i) == read(&b_out, start);
        if !same || i == N - 1 {
            let end = if i == N - 1 && same { N } else { i };
            let sum: f32 = a[start..end].iter().sum();
            assert_eq!(read(&b_out, start), sum, "reduce at {start}..{end}");
            assert_eq!(read(&c_out, start), a[start], "broadcast at {start}..{end}");
            for j in start..end {
                let expect = a[start + ((j - start + 1) % 4) % (end - start)];
                assert_eq!(read(&d_out, j), expect, "shuffle at {j}");
            }
            subgroups += 1;
            start = end;
        }
    }
    println!(
        "subgroup ok on {} ({} subgroups detected)",
        device.name(),
        subgroups
    );
}

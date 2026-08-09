use saturn_api::{BackendKind, open};
use saturn_core::{BindingRef, BufferSpec, KernelSpec};

const N: usize = 64;

fn main() {
    let device: Box<dyn saturn_core::Device> = match std::env::args().nth(1).as_deref() {
        Some("metal") => open(BackendKind::Metal).expect("open metal"),
        Some("vulkan-validation") => Box::new(
            saturn_vk::VkDevice::open(&saturn_vk::Options { validation: true })
                .expect("open vulkan with validation"),
        ),
        _ => open(BackendKind::Vulkan).expect("open vulkan"),
    };

    let src = r#"
fn clamp_sat(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo { return lo; }
    if v > hi { return hi; }
    return v;
}
fn apply_gain(v: f32, const GAIN: f32) -> f32 {
    return v * GAIN;
}
fn count_set(w: u32) -> u32 {
    return popcount(w) + clz(w) + ctz(w);
}
kernel lf [workgroup(64, 1, 1)]
    (src: buf<f32>, dst: buf<f32>, bits: buf<u32>, acc: buf<u32>)
{
    spec MODE: u32 = 0;
    let lo = -1;
    let v = src[gid.x];
    if MODE == 1 {
        dst[gid.x] = clamp_sat(apply_gain(v, 2.0), lo as f32, 1.0);
    } else {
        dst[gid.x] = trunc(v) + sign(v) + fract(v);
    }
    let vv = vec4<f32>(v, v, v, v);
    dst[gid.x] = dst[gid.x] + dot(vv, vv);
    bits[gid.x] = count_set(bits[gid.x]);
    atomic_add(acc, 0, bits[gid.x]);
    atomic_and(acc, 1, bits[gid.x] & 255);
    atomic_or(acc, 2, 1);
    atomic_xor(acc, 3, bits[gid.x]);
}
"#;
    let kernel = device
        .create_kernel(&KernelSpec::new("lang", src).with_specs(&[("MODE", 1.0)]))
        .expect("create kernel");

    let mut src_data = vec![0f32; N];
    for i in 0..N {
        src_data[i] = (i as f32) / 16.0 - 1.0;
    }
    let mut bits_data = vec![0u32; N];
    for i in 0..N {
        bits_data[i] = (i as u32) * 7;
    }
    let src_buf = device
        .create_buffer(&BufferSpec {
            size: (N * 4) as u64,
            host_visible: true,
        })
        .expect("create src");
    let dst_buf = device
        .create_buffer(&BufferSpec {
            size: (N * 4) as u64,
            host_visible: true,
        })
        .expect("create dst");
    let bits_buf = device
        .create_buffer(&BufferSpec {
            size: (N * 4) as u64,
            host_visible: true,
        })
        .expect("create bits");
    let acc_buf = device
        .create_buffer(&BufferSpec {
            size: 4 * 4 * 4,
            host_visible: true,
        })
        .expect("create acc");
    src_buf.write(0, &encode_f32(&src_data)).expect("write src");
    bits_buf.write(0, &encode_u32(&bits_data)).expect("write bits");
    acc_buf.write(0, &[0u8; 16]).expect("write acc");

    let mut encoder = device.encoder().expect("encoder");
    encoder
        .bind(
            &*kernel,
            &[
                BindingRef {
                    index: 0,
                    buffer: &*src_buf,
                    offset: 0,
                    size: 0,
                },
                BindingRef {
                    index: 1,
                    buffer: &*dst_buf,
                    offset: 0,
                    size: 0,
                },
                BindingRef {
                    index: 2,
                    buffer: &*bits_buf,
                    offset: 0,
                    size: 0,
                },
                BindingRef {
                    index: 3,
                    buffer: &*acc_buf,
                    offset: 0,
                    size: 0,
                },
            ],
        )
        .expect("bind");
    encoder
        .dispatch([1, 1, 1])
        .expect("dispatch");
    let submission = device.submit(encoder).expect("submit");
    submission.wait().expect("wait");

    let mut dst = vec![0u8; N * 4];
    let mut bits = vec![0u8; N * 4];
    let mut acc = vec![0u8; 16];
    dst_buf.read(0, &mut dst).expect("read dst");
    bits_buf.read(0, &mut bits).expect("read bits");
    acc_buf.read(0, &mut acc).expect("read acc");
    let dst = decode_f32(&dst);
    let bits = decode_u32(&bits);
    let acc = decode_u32(&acc);
    let mut max_err = 0.0f32;
    for i in 0..N {
        let v = src_data[i];
        let gain = (v * 2.0).clamp(-1.0, 1.0);
        let dot = v * v * 4.0;
        let expect = gain + dot;
        max_err = max_err.max((dst[i] - expect).abs());
    }
    let mut pop_sum = 0u32;
    let mut pop_xor = 0u32;
    for i in 0..N {
        let w = bits_data[i];
        let expect = w.count_ones() + w.leading_zeros() + w.trailing_zeros();
        assert_eq!(bits[i], expect, "bit ops mismatch at {i}");
        pop_sum += expect;
        pop_xor ^= expect;
    }
    assert_eq!(acc[0], pop_sum, "atomic_add mismatch");
    assert_eq!(acc[1], 0, "atomic_and mismatch");
    assert_eq!(acc[2], 1, "atomic_or mismatch");
    assert_eq!(acc[3], pop_xor, "atomic_xor mismatch");
    assert!(
        max_err < 1e-3,
        "kernel output mismatch, max_err={max_err}"
    );
    println!(
        "lang ok on {} (spec MODE=1, fn+dot+bitops+atomics, max_err {max_err})",
        device.name()
    );
}

fn encode_f32(data: &[f32]) -> Vec<u8> {
    data.iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

fn encode_u32(data: &[u32]) -> Vec<u8> {
    data.iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

fn decode_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn decode_u32(data: &[u8]) -> Vec<u32> {
    data.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

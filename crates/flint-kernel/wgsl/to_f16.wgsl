enable f16;

struct Pc {
    N_ELEM: u32,
}
var<immediate> pc: Pc;

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> xf: array<f16>;

@compute @workgroup_size(256, 1, 1)
fn to_f16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x * 4u;
    let v = vec4<f32>(x[i], x[i + 1u], x[i + 2u], x[i + 3u]);
    xf[i] = f16(v.x);
    xf[i + 1u] = f16(v.y);
    xf[i + 2u] = f16(v.z);
    xf[i + 3u] = f16(v.w);
}

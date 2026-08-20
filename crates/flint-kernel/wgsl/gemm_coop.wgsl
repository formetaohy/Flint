@compute @workgroup_size(256, 1, 1)
fn gemm_coop(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) grid: vec3<u32>,
) {
    let N = pc.N;
    let K = pc.K;
    let WDTYPE = pc.WDTYPE;
    let GROUP = pc.GROUP;
    let ACC = pc.ACC;
    let Y_STRIDE = pc.Y_STRIDE;
    let Y_OFF = pc.Y_OFF;
    let m0 = grid.y * TM;
    let n0 = grid.x * TN;
    let steps = K / BK;
    let sg = lid.x / 32u;
    let sr = sg / 2u;
    let sc = sg % 2u;
    let r0 = m0 + sr * 32u;
    let c0 = n0 + sc * 64u;

    stage_b(0u, 0u, n0, lid.x, N, K, WDTYPE, GROUP);
    workgroupBarrier();

    let y0 = r0 * Y_STRIDE + Y_OFF + c0;
    let y1 = y0 + 16u;
    let y2 = y0 + 32u;
    let y3 = y0 + 48u;
    let y4 = y0 + 16u * Y_STRIDE;
    let y5 = y4 + 16u;
    let y6 = y4 + 32u;
    let y7 = y4 + 48u;

    var acc00: coop_mat16x16<f32, C>;
    var acc01: coop_mat16x16<f32, C>;
    var acc02: coop_mat16x16<f32, C>;
    var acc03: coop_mat16x16<f32, C>;
    var acc10: coop_mat16x16<f32, C>;
    var acc11: coop_mat16x16<f32, C>;
    var acc12: coop_mat16x16<f32, C>;
    var acc13: coop_mat16x16<f32, C>;
    if ACC == 1u {
        acc00 = coopLoadT<coop_mat16x16<f32, C>>(&y[y0], Y_STRIDE);
        acc01 = coopLoadT<coop_mat16x16<f32, C>>(&y[y1], Y_STRIDE);
        acc02 = coopLoadT<coop_mat16x16<f32, C>>(&y[y2], Y_STRIDE);
        acc03 = coopLoadT<coop_mat16x16<f32, C>>(&y[y3], Y_STRIDE);
        acc10 = coopLoadT<coop_mat16x16<f32, C>>(&y[y4], Y_STRIDE);
        acc11 = coopLoadT<coop_mat16x16<f32, C>>(&y[y5], Y_STRIDE);
        acc12 = coopLoadT<coop_mat16x16<f32, C>>(&y[y6], Y_STRIDE);
        acc13 = coopLoadT<coop_mat16x16<f32, C>>(&y[y7], Y_STRIDE);
    } else {
        acc00 = coop_mat16x16<f32, C>();
        acc01 = coop_mat16x16<f32, C>();
        acc02 = coop_mat16x16<f32, C>();
        acc03 = coop_mat16x16<f32, C>();
        acc10 = coop_mat16x16<f32, C>();
        acc11 = coop_mat16x16<f32, C>();
        acc12 = coop_mat16x16<f32, C>();
        acc13 = coop_mat16x16<f32, C>();
    }

    for (var it = 0u; it < steps; it++) {
        let p = it % 2u;
        let p1 = 1u - p;
        workgroupBarrier();
        if it + 1u < steps {
            stage_b(p1, (it + 1u) * BK, n0, lid.x, N, K, WDTYPE, GROUP);
        }
        let kb = it * BK;
        let ab = r0 * K + kb;
        let a0 = ab;
        let a1 = ab + 16u;
        let a2 = ab + 16u * K;
        let a3 = a2 + 16u;
        let bb = p * (TN * BK) + sc * 64u * BK;
        let b0 = bb;
        let b1 = bb + 16u * BK;
        let b2 = bb + 32u * BK;
        let b3 = bb + 48u * BK;
        let b4 = bb + 16u;
        let b5 = b1 + 16u;
        let b6 = b2 + 16u;
        let b7 = b3 + 16u;
        let x00 = coopLoadT<coop_mat16x16<f16, A>>(&xf[a0], K);
        let x01 = coopLoadT<coop_mat16x16<f16, A>>(&xf[a2], K);
        let x10 = coopLoadT<coop_mat16x16<f16, A>>(&xf[a1], K);
        let x11 = coopLoadT<coop_mat16x16<f16, A>>(&xf[a3], K);
        let w00 = coopLoad<coop_mat16x16<f16, B>>(&ws[b0], BK);
        let w01 = coopLoad<coop_mat16x16<f16, B>>(&ws[b1], BK);
        let w02 = coopLoad<coop_mat16x16<f16, B>>(&ws[b2], BK);
        let w03 = coopLoad<coop_mat16x16<f16, B>>(&ws[b3], BK);
        let w10 = coopLoad<coop_mat16x16<f16, B>>(&ws[b4], BK);
        let w11 = coopLoad<coop_mat16x16<f16, B>>(&ws[b5], BK);
        let w12 = coopLoad<coop_mat16x16<f16, B>>(&ws[b6], BK);
        let w13 = coopLoad<coop_mat16x16<f16, B>>(&ws[b7], BK);
        acc00 = coopMultiplyAdd(x00, w00, acc00);
        acc01 = coopMultiplyAdd(x00, w01, acc01);
        acc02 = coopMultiplyAdd(x00, w02, acc02);
        acc03 = coopMultiplyAdd(x00, w03, acc03);
        acc10 = coopMultiplyAdd(x01, w00, acc10);
        acc11 = coopMultiplyAdd(x01, w01, acc11);
        acc12 = coopMultiplyAdd(x01, w02, acc12);
        acc13 = coopMultiplyAdd(x01, w03, acc13);
        acc00 = coopMultiplyAdd(x10, w10, acc00);
        acc01 = coopMultiplyAdd(x10, w11, acc01);
        acc02 = coopMultiplyAdd(x10, w12, acc02);
        acc03 = coopMultiplyAdd(x10, w13, acc03);
        acc10 = coopMultiplyAdd(x11, w10, acc10);
        acc11 = coopMultiplyAdd(x11, w11, acc11);
        acc12 = coopMultiplyAdd(x11, w12, acc12);
        acc13 = coopMultiplyAdd(x11, w13, acc13);
        workgroupBarrier();
    }

    coopStoreT(acc00, &y[y0], Y_STRIDE);
    coopStoreT(acc01, &y[y1], Y_STRIDE);
    coopStoreT(acc02, &y[y2], Y_STRIDE);
    coopStoreT(acc03, &y[y3], Y_STRIDE);
    coopStoreT(acc10, &y[y4], Y_STRIDE);
    coopStoreT(acc11, &y[y5], Y_STRIDE);
    coopStoreT(acc12, &y[y6], Y_STRIDE);
    coopStoreT(acc13, &y[y7], Y_STRIDE);
}

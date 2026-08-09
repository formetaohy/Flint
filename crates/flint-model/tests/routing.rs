use flint_model::routing::{RouteKind, Routing};

#[test]
fn softmax_topk_weights_sum_lt_one() {
    let logits: Vec<f32> = vec![
        2.0, 1.0, 0.5, -1.0, 3.0, 0.0, -2.0, 1.5, 0.25, -0.5, 2.5, 1.0, 0.0, -3.0, 4.0, 0.75,
    ];
    let r = Routing::new(&logits, 1, 16, 2, RouteKind::Softmax, 0.0);
    assert_eq!(r.count(14), 1, "expert 14 (logit 4.0) is top-1");
    assert_eq!(r.count(4), 1, "expert 4 (logit 3.0) is top-2");
    assert_eq!(r.count(0), 0);
    let w: f32 = r.weights.iter().sum();
    assert!(w < 1.0 && w > 0.5, "top-2 softmax weights sum to {w}");
}

#[test]
fn sparsemixer_masks_distant_experts() {

    let logits: Vec<f32> = vec![
        10.0, 9.9, 5.0, -5.0, 9.95, 9.98, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 9.99,
    ];
    let r = Routing::new(
        &logits,
        1,
        16,
        2,
        RouteKind::SparseMixer { jitter: 0.01 },
        0.0,
    );
    let mut hits = Vec::new();
    for e in 0..16 {
        if r.count(e) > 0 {
            hits.push(e);
        }
    }
    assert!(hits.contains(&0), "top-1 expert 0 selected");
    assert!(
        hits.len() == 2,
        "exactly two experts selected, got {hits:?}"
    );

    let max_w = r.weights.iter().cloned().fold(0.0f32, f32::max);
    assert!(max_w > 0.2, "near-max softmax weight {max_w}");
}

#[test]
fn shared_expert_covers_every_row() {
    let logits: Vec<f32> = (0..64).map(|i| (i % 7) as f32).collect();
    let r = Routing::new(&logits, 4, 4, 2, RouteKind::Softmax, 0.5);
    assert_eq!(r.count(4), 4, "virtual shared expert covers all rows");

    for e in 0..=4 {
        assert_eq!(r.starts[e] % 64, 0, "expert {e} start aligned");
    }
    assert_eq!(r.rows.len(), r.starts[5] as usize);
    assert_eq!(r.rows[r.starts[4] as usize], 0, "shared row 0 at its slot");
    assert_eq!(r.weights[r.starts[4] as usize], 0.5);
    assert_eq!(r.weights[r.starts[4] as usize + 3], 0.5);
}

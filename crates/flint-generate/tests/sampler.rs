use flint_generate::{Dist, Sampler, SamplingParams, apply_repeat_penalty, softmax};

fn params() -> SamplingParams {
    SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        ..Default::default()
    }
}

fn probs(d: &Dist) -> &[f32] {
    match d {
        Dist::Probs(p) => p,
        Dist::Greedy(_) => panic!("expected a stochastic distribution"),
    }
}

fn logits(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u32 << 31) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

#[test]
fn greedy_transform_is_penalized_argmax() {
    let s = Sampler::greedy(1);
    let logits = [0.1f32, 5.0, -2.0, 4.9];
    assert_eq!(s.transform(&logits, &[], None), Dist::Greedy(1));

    let penalized = Sampler::new(
        SamplingParams {
            temperature: 0.0,
            repeat_penalty: 10.0,
            ..Default::default()
        },
        1,
    );
    assert_eq!(penalized.transform(&logits, &[1, 1, 1], None), Dist::Greedy(3));
}

#[test]
fn stochastic_transform_renormalizes_over_the_kept_set() {
    let logits: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let s = Sampler::new(
        SamplingParams {
            top_k: 3,
            ..params()
        },
        0,
    );
    let d = s.transform(&logits, &[], None);
    let p = probs(&d);
    assert_eq!(p.len(), 10);
    let nonzero: Vec<u32> = p
        .iter()
        .enumerate()
        .filter(|(_, v)| **v > 0.0)
        .map(|(i, _)| i as u32)
        .collect();
    assert_eq!(
        nonzero,
        vec![7, 8, 9],
        "top_k=3 keeps exactly the top three"
    );
    assert!(
        (p.iter().sum::<f32>() - 1.0).abs() < 1e-5,
        "renormalized to sum 1"
    );
}

#[test]
fn top_p_keeps_only_the_nucleus() {
    let logits: Vec<f32> = (0..8).rev().map(|i| i as f32).collect();
    let raw = softmax(&logits, 1.0);
    let mut acc = 0.0f32;
    let mut nucleus = Vec::new();
    for (i, p) in raw.iter().enumerate() {
        nucleus.push(i);
        acc += *p;
        if acc >= 0.6 {
            break;
        }
    }
    assert!(nucleus.len() < 8, "the nucleus must be a proper subset");

    let s = Sampler::new(
        SamplingParams {
            top_p: 0.6,
            ..params()
        },
        0,
    );
    let d = s.transform(&logits, &[], None);
    let p = probs(&d);
    for (i, &v) in p.iter().enumerate() {
        assert!(
            v == 0.0 || nucleus.contains(&i),
            "top_p=0.6 kept non-nucleus token {i}"
        );
    }
}

#[test]
fn min_p_drops_tokens_below_the_relative_floor() {
    let logits = [10.0f32, 0.0, 0.0, 0.0];
    let s = Sampler::new(
        SamplingParams {
            min_p: 0.5,
            ..params()
        },
        0,
    );
    let d = s.transform(&logits, &[], None);
    let p = probs(&d);
    assert!(p[0] > 0.99, "dominant token survives");
    assert!(
        p[1..].iter().all(|&v| v == 0.0),
        "tokens below the floor are zeroed"
    );
}

#[test]
fn draw_is_deterministic_per_seed() {
    let logits = logits(50, 7, 2.0);
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 10,
        top_p: 0.9,
        min_p: 0.02,
        ..Default::default()
    };
    let mut a = Sampler::new(params, 99);
    let mut b = Sampler::new(params, 99);
    let da = a.transform(&logits, &[], None);
    let db = b.transform(&logits, &[], None);
    for _ in 0..20 {
        assert_eq!(a.draw(&da), b.draw(&db));
    }
}

#[test]
fn repeat_penalty_is_symmetric_around_zero() {
    let mut scores = [2.0f32, -2.0, 1.0];
    apply_repeat_penalty(&mut scores, &[0, 1], 2.0, 8);
    assert_eq!(scores[0], 1.0, "positive logit divided");
    assert_eq!(scores[1], -4.0, "negative logit multiplied");
    assert_eq!(scores[2], 1.0, "unseen logit untouched");
}

#[test]
fn repeat_penalty_respects_last_n_and_disable() {
    let mut scores = [2.0f32, 0.5];
    apply_repeat_penalty(&mut scores, &[0, 1, 1, 1], 2.0, 2);
    assert_eq!(scores[0], 2.0, "id 0 fell out of the 2-token window");
    assert_eq!(
        scores[1], 0.125,
        "id 1 appears twice in the window: penalties stack"
    );

    let mut disabled = [2.0f32, -2.0];
    apply_repeat_penalty(&mut disabled, &[0, 1], 1.0, 8);
    assert_eq!(disabled, [2.0, -2.0], "penalty 1.0 is a no-op");

    let mut zero_window = [2.0f32, 1.0];
    apply_repeat_penalty(&mut zero_window, &[0, 1], 2.0, 0);
    assert_eq!(zero_window, [2.0, 1.0], "last_n=0 sees no context");
}

#[test]
fn default_params_match_common_chat_inference() {
    let d = SamplingParams::default();
    assert_eq!(d.temperature, 0.7);
    assert_eq!(d.top_k, 20);
    assert_eq!(d.top_p, 0.8);
    assert_eq!(d.min_p, 0.0);
    assert_eq!(d.repeat_penalty, 1.0);
    assert_eq!(d.repeat_last_n, 64);
}

#[test]
fn softmax_sums_to_one_and_sharpens() {
    let logits = [1.0f32, 2.0, 3.0];
    let p = softmax(&logits, 1.0);
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    let cold = softmax(&logits, 0.1);
    assert!(
        cold[2] > p[2],
        "lower temperature concentrates mass on the max"
    );

    let shifted = softmax(&[1001.0f32, 1002.0, 1003.0], 1.0);
    assert!(
        (shifted[2] - p[2]).abs() < 1e-5,
        "max subtraction keeps large logits stable"
    );
}

fn transformed(s: &Sampler, logits: &[f32]) -> Dist {
    s.transform(logits, &[], None)
}

#[test]
fn greedy_verify_accepts_only_argmax() {
    let mut s = Sampler::greedy(0);
    let target = transformed(&s, &[0.0f32, 3.0, 1.0]);
    let draft = transformed(&s, &[0.0f32, 0.0, 0.0]);
    assert_eq!(s.verify(&target, &draft, 1), (true, 1));
    assert_eq!(
        s.verify(&target, &draft, 2),
        (false, 1),
        "reject resamples the target argmax"
    );
}

#[test]
fn verify_accepts_whenever_target_covers_draft() {
    let mut s = Sampler::new(params(), 3);
    let target = transformed(&s, &[5.0f32, 0.0, 0.0]);
    let draft = transformed(&s, &[1.0f32, 0.0, 0.0]);
    for _ in 0..10 {
        assert_eq!(
            s.verify(&target, &draft, 0),
            (true, 0),
            "target mass dominates the draft"
        );
    }
}

#[test]
fn verify_rejects_zero_target_mass_and_resamples_the_residual() {
    let mut s = Sampler::new(params(), 4);
    let target = transformed(&s, &[0.0f32, -1000.0, 0.0]);
    let draft = transformed(&s, &[0.0f32, 0.0, 0.0]);
    for _ in 0..10 {
        let (ok, tok) = s.verify(&target, &draft, 1);
        assert!(!ok, "zero target mass must reject");
        assert!(tok == 0 || tok == 2, "residual drew token {tok}");
    }
}

#[test]
fn verify_output_distribution_equals_target() {
    const VOCAB: usize = 64;
    const N: usize = 100_000;
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 12,
        top_p: 0.9,
        min_p: 0.01,
        repeat_penalty: 1.3,
        repeat_last_n: 8,
    };
    let context: Vec<u32> = vec![3, 7, 7, 12, 40];
    let target_logits = logits(VOCAB, 11, 3.0);
    let draft_logits = logits(VOCAB, 29, 3.0);

    let mut s = Sampler::new(params, 1234);
    let pt = s.transform(&target_logits, &context, None);
    let pd = s.transform(&draft_logits, &context, None);
    let pt = probs(&pt);
    let pd = probs(&pd);

    let mut counts = vec![0usize; VOCAB];
    let mut accepted = 0usize;
    for _ in 0..N {
        let d = s.draw(&Dist::Probs(pd.to_vec()));
        let (ok, tok) = s.verify(&Dist::Probs(pt.to_vec()), &Dist::Probs(pd.to_vec()), d);
        accepted += ok as usize;
        counts[tok as usize] += 1;
    }

    let mut chi2 = 0.0f64;
    let mut df = 0;
    let mut max_dev = 0.0f32;
    for (i, &p) in pt.iter().enumerate() {
        let exp = p * N as f32;
        let obs = counts[i] as f32;
        max_dev = max_dev.max((obs / N as f32 - p).abs());
        if exp >= 50.0 {
            chi2 += ((obs - exp) * (obs - exp) / exp) as f64;
            df += 1;
        }
    }
    df -= 1;

    assert!(
        chi2 < (df as f64) * 3.0 + 50.0,
        "committed tokens do not follow the target distribution: chi2 {chi2:.1} at df {df}"
    );
    assert!(
        max_dev < 0.01,
        "empirical frequency deviates {max_dev} from target probability"
    );

    let theoretical: f32 = pt.iter().zip(pd).map(|(&a, &b)| a.min(b)).sum();
    let empirical = accepted as f32 / N as f32;
    assert!(
        (empirical - theoretical).abs() < 0.01,
        "acceptance rate {empirical} vs theoretical {theoretical}"
    );
}

#[test]
fn greedy_spec_sequence_matches_plain_greedy() {
    const STEPS: usize = 64;
    let target = (0..STEPS)
        .map(|i| logits(32, 1000 + i as u64, 2.0))
        .collect::<Vec<_>>();
    let draft = (0..STEPS)
        .map(|i| logits(32, 5000 + i as u64, 2.0))
        .collect::<Vec<_>>();

    let mut plain = Sampler::greedy(9);
    let plain_tokens: Vec<u32> = target
        .iter()
        .map(|l| plain.draw(&plain.transform(l, &[], None)))
        .collect();

    let mut spec = Sampler::greedy(9);
    let mut spec_tokens = Vec::new();
    for i in 0..STEPS {
        let pd = spec.transform(&draft[i], &[], None);
        let d = spec.draw(&pd);
        let pt = spec.transform(&target[i], &[], None);
        let (_, tok) = spec.verify(&pt, &pd, d);
        spec_tokens.push(tok);
    }
    assert_eq!(spec_tokens, plain_tokens);
}

#[test]
fn mask_zeroes_banned_tokens_in_both_modes() {
    let logits = [0.1f32, 5.0, -2.0, 4.9];
    let mask = [1.0, 0.0, 1.0, 0.0];
    let s = Sampler::greedy(1);
    assert_eq!(s.transform(&logits, &[], Some(&mask)), Dist::Greedy(0));
    let st = Sampler::new(params(), 1);
    let d = st.transform(&logits, &[], Some(&mask));
    let p = probs(&d);
    assert_eq!(p[1], 0.0);
    assert_eq!(p[3], 0.0);
    assert!(p[0] > 0.0 && p[2] > 0.0);
}

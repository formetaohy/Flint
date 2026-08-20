use flint_model::pool::{ArenaSpec, KvArena};

#[test]
fn alloc_grows_and_reuses_pages() {
    let mut arena = KvArena::new(&ArenaSpec {
        seq_lens: vec![64, 64],
        pages: None,
    })
    .unwrap();
    assert_eq!(arena.pages(), 4);
    assert_eq!(arena.used(), 0);

    arena.alloc(0, 0, 40).unwrap();
    assert_eq!(arena.used(), 2);
    assert_eq!(arena.table_of(0), &[0, 1]);

    arena.alloc(0, 40, 24).unwrap();
    assert_eq!(arena.used(), 2);
    assert_eq!(arena.table_of(0).len(), 2);

    arena.free_seq(0);
    assert_eq!(arena.used(), 0);
    arena.alloc(1, 0, 64).unwrap();
    assert_eq!(arena.used(), 2);
    assert_eq!(arena.table_of(1).len(), 2);
}

#[test]
fn truncate_keeps_prefix_and_frees_tail() {
    let mut arena = KvArena::new(&ArenaSpec {
        seq_lens: vec![128],
        pages: None,
    })
    .unwrap();
    arena.alloc(0, 0, 128).unwrap();
    assert_eq!(arena.table_of(0).len(), 4);

    arena.truncate(0, 70);
    assert_eq!(arena.table_of(0).len(), 3);
    assert_eq!(arena.used(), 3);

    arena.alloc(0, 70, 30).unwrap();
    assert_eq!(arena.table_of(0).len(), 4);
    assert_eq!(arena.used(), 4);
}

#[test]
fn free_seq_returns_every_page() {
    let mut arena = KvArena::new(&ArenaSpec {
        seq_lens: vec![96, 96],
        pages: Some(3),
    })
    .unwrap();
    arena.alloc(0, 0, 96).unwrap();
    assert_eq!(arena.used(), 3);
    assert!(arena.alloc(1, 0, 32).is_err());

    arena.free_seq(0);
    arena.alloc(1, 0, 96).unwrap();
    assert_eq!(arena.table_of(1).len(), 3);
    assert_eq!(arena.used(), 3);
}

#[test]
fn exhaustion_fails_fast() {
    let mut arena = KvArena::new(&ArenaSpec {
        seq_lens: vec![128],
        pages: Some(2),
    })
    .unwrap();
    assert!(arena.alloc(0, 0, 128).is_err());
    arena.alloc(0, 0, 64).unwrap();
    assert_eq!(arena.used(), 2);
}

#[test]
fn context_limit_is_enforced() {
    let mut arena = KvArena::new(&ArenaSpec {
        seq_lens: vec![32],
        pages: None,
    })
    .unwrap();
    arena.alloc(0, 0, 32).unwrap();
    let err = arena.alloc(0, 32, 1).err().unwrap();
    assert!(err.to_string().contains("context limit 32"), "{err}");
}

#[test]
fn table_flattens_with_sentinels() {
    let mut arena = KvArena::new(&ArenaSpec {
        seq_lens: vec![32, 64],
        pages: Some(4),
    })
    .unwrap();
    arena.alloc(1, 0, 40).unwrap();
    let table = arena.table();
    assert_eq!(table.len(), 4, "two seqs of up to two pages");
    assert_eq!(
        &table[0..2],
        &[u32::MAX, u32::MAX],
        "empty seq is sentinels"
    );
    assert_ne!(table[2], u32::MAX);
    assert_ne!(table[3], u32::MAX);
    arena.alloc(0, 0, 32).unwrap();
    let table = arena.table();
    assert_ne!(table[0], u32::MAX);
    assert_ne!(table[0], table[2]);
}

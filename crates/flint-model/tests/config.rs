//! Config parsing helpers: typed field access and dimension validation.

use flint_model::config::{
    bool_field, check_gemm_dims, check_head_dim, f64_field, req, u32_field, u32_list,
};
use serde_json::json;

#[test]
fn typed_field_access() {
    let v = json!({"n": 42u32, "f": 1.5, "b": true, "s": "x", "neg": -3});

    assert_eq!(req(&v, "n").unwrap(), &json!(42));
    assert!(req(&v, "missing").is_err());

    assert_eq!(u32_field(&v, "n").unwrap(), 42);
    assert!(u32_field(&v, "missing").is_err());
    assert!(u32_field(&v, "s").is_err(), "string is not a u32");
    assert!(u32_field(&v, "neg").is_err(), "negative is not a u32");
    assert!(u32_field(&v, "f").is_err(), "fractional is not a u32");

    assert_eq!(f64_field(&v, "f").unwrap(), 1.5);
    assert_eq!(f64_field(&v, "n").unwrap(), 42.0, "integers coerce to f64");
    assert!(f64_field(&v, "s").is_err());

    assert!(bool_field(&v, "b").unwrap());
    assert!(bool_field(&v, "n").is_err());
}

#[test]
fn u32_list_accepts_absent_single_and_array() {
    let v =
        json!({"one": 7u32, "many": [1u32, 2, 3], "nil": null, "bad": "x", "bad_arr": [1, "y"]});

    assert_eq!(u32_list(&v, "missing").unwrap(), Vec::<u32>::new());
    assert_eq!(u32_list(&v, "nil").unwrap(), Vec::<u32>::new());
    assert_eq!(u32_list(&v, "one").unwrap(), vec![7]);
    assert_eq!(u32_list(&v, "many").unwrap(), vec![1, 2, 3]);
    assert!(u32_list(&v, "bad").is_err());
    assert!(u32_list(&v, "bad_arr").is_err());
}

#[test]
fn gemm_dims_must_tile_to_16() {
    assert!(check_gemm_dims(&[(16, 64), (256, 128), (49152, 960)]).is_ok());
    assert!(check_gemm_dims(&[(15, 64)]).is_err(), "N must be a multiple of 16");
    assert!(check_gemm_dims(&[(16, 32)]).is_err(), "K must be a multiple of 64");
}

#[test]
fn head_dim_bounds() {
    assert!(check_head_dim(64).is_ok());
    assert!(check_head_dim(256).is_ok());
    assert!(check_head_dim(512).is_ok(), "Gemma 4 global heads widen to 512");
    assert!(check_head_dim(63).is_err());
    assert!(check_head_dim(513).is_err());
}

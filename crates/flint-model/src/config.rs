use flint_error::{Error, Result};
use serde_json::Value;

pub fn req<'a>(v: &'a Value, key: &str) -> Result<&'a Value> {
    v.get(key)
        .ok_or_else(|| Error::Config(format!("missing config field {key:?}")))
}

pub fn u32_field(v: &Value, key: &str) -> Result<u32> {
    req(v, key)?
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| Error::Config(format!("field {key:?} is not a u32")))
}

pub fn f64_field(v: &Value, key: &str) -> Result<f64> {
    req(v, key)?
        .as_f64()
        .ok_or_else(|| Error::Config(format!("field {key:?} is not a number")))
}

pub fn bool_field(v: &Value, key: &str) -> Result<bool> {
    req(v, key)?
        .as_bool()
        .ok_or_else(|| Error::Config(format!("field {key:?} is not a bool")))
}

pub fn u32_list(v: &Value, key: &str) -> Result<Vec<u32>> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Number(n)) => Ok(vec![as_u32(n.as_u64(), key)?]),
        Some(Value::Array(items)) => items.iter().map(|it| as_u32(it.as_u64(), key)).collect(),
        Some(_) => Err(Error::Config(format!("field {key:?} is not an id list"))),
    }
}

fn as_u32(n: Option<u64>, key: &str) -> Result<u32> {
    n.and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| Error::Config(format!("field {key:?} is not a u32")))
}

pub fn check_gemm_dims(pairs: &[(u32, u32)]) -> Result<()> {
    for &(n, k) in pairs {
        if !n.is_multiple_of(16) || !k.is_multiple_of(64) {
            return Err(Error::Config(format!(
                "dimension pair (N={n}, K={k}) does not satisfy N%16 and K%64"
            )));
        }
    }
    Ok(())
}

pub fn check_head_dim(head_dim: u32) -> Result<()> {
    if !(64..=512).contains(&head_dim) {
        return Err(Error::Config(format!(
            "head_dim {head_dim} outside [64, 512]"
        )));
    }
    Ok(())
}

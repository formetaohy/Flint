use thuban_error::{Error, Result};
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

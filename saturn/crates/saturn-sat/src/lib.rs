use std::path::{Path, PathBuf};
use std::str::FromStr;

use proc_macro::TokenStream;
use syn::LitStr;
use syn::parse::Parse;
use syn::parse_macro_input;

use saturn_compiler::Driver;
use saturn_core::Scalar;

fn sat_path(call_file: &Path, name: &str) -> Result<PathBuf, String> {
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        return Err(format!(
            "sat file name must be a plain file name, got '{name}'"
        ));
    }
    let mut dir = call_file
        .parent()
        .ok_or_else(|| format!("cannot resolve directory of {}", call_file.display()))?;
    loop {
        let candidate = dir.join("sat").join(name);
        if candidate.is_file() {
            return std::path::absolute(&candidate)
                .map_err(|e| format!("cannot resolve {}: {e}", candidate.display()));
        }
        dir = dir
            .parent()
            .ok_or_else(|| format!("cannot find sat/{name} from {}", call_file.display()))?;
    }
}

fn scalar_ts(scalar: Scalar) -> &'static str {
    match scalar {
        Scalar::F32 => "::saturn_core::Scalar::F32",
        Scalar::F16 => "::saturn_core::Scalar::F16",
        Scalar::Bf16 => "::saturn_core::Scalar::Bf16",
        Scalar::I32 => "::saturn_core::Scalar::I32",
        Scalar::U32 => "::saturn_core::Scalar::U32",
        Scalar::I8 => "::saturn_core::Scalar::I8",
        Scalar::U8 => "::saturn_core::Scalar::U8",
        Scalar::Bool => "::saturn_core::Scalar::Bool",
    }
}

fn precompile(path: &Path) -> Result<String, String> {
    let source = saturn_compiler::Source::load(path).map_err(|e| e.to_string())?;
    let kernel = Driver::new().compile(&source).map_err(|diags| {
        diags
            .iter()
            .map(|d| format!("{}: {}", source.render_span(d.span), d.msg))
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    let spirv = saturn_shader::to_spirv(&kernel).map_err(|e| format!("SPIR-V generation: {e}"))?;
    let (msl, entry) = saturn_shader::to_msl(&kernel).map_err(|e| format!("MSL generation: {e}"))?;
    if kernel.name != entry {
        return Err(format!(
            "kernel entry mismatch: {} vs {entry}",
            kernel.name
        ));
    }
    let scalars = kernel
        .scalars
        .iter()
        .map(|p| {
            format!(
                "::saturn_core::PrecompiledScalar {{ name: {:?}, offset: {}, ty: {} }}",
                p.name,
                p.offset,
                scalar_ts(p.ty)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let triples = kernel
        .coop_triples
        .iter()
        .map(|(a, b, c)| {
            format!(
                "({}, {}, {})",
                scalar_ts(*a),
                scalar_ts(*b),
                scalar_ts(*c)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let roles = kernel
        .coop_roles
        .iter()
        .map(|(elem, role)| {
            format!(
                "({}, {})",
                scalar_ts(*elem),
                role.encode()
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let spirv_ts = spirv
        .iter()
        .map(|b| b.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let msl_ts = format!("{msl:?}");
    let wg = kernel.workgroup_size;
    let ts = format!(
        "&::saturn_core::PrecompiledKernel {{
             name: {:?},
             workgroup_size: [{}, {}, {}],
             buffers: {},
             spirv: &[{spirv_ts}],
             msl: {msl_ts},
             scalars: &[{scalars}],
             coop_triples: &[{triples}],
             coop_roles: &[{roles}],
         }}",
        kernel.name,
        wg[0],
        wg[1],
        wg[2],
        kernel.params.len(),
    );
    ts.parse().map_err(|e| format!("macro output generation: {e}"))
}

struct SatName(String);

impl Parse for SatName {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let lit = input.parse::<LitStr>()?;
        if lit.value().contains('/') || lit.value().contains('\\') || lit.value().contains("..") {
            return Err(syn::Error::new(
                lit.span(),
                format!("expected a plain sat file name, got '{}'", lit.value()),
            ));
        }
        Ok(SatName(lit.value()))
    }
}

fn compile_error(msg: String) -> TokenStream {
    format!("::core::compile_error!({msg:?})").parse().unwrap()
}

#[proc_macro]
pub fn sat(input: TokenStream) -> TokenStream {
    let name = parse_macro_input!(input as SatName).0;
    let Some(call_file) = proc_macro::Span::call_site().local_file() else {
        return compile_error("cannot resolve macro call site file".to_string());
    };
    match sat_path(&call_file, &name) {
        Ok(path) => match precompile(&path) {
            Ok(ts) => TokenStream::from_str(ts.as_str())
                .unwrap_or_else(|e| compile_error(format!("sat/{name} output generation: {e}"))),
            Err(msg) => compile_error(format!("sat/{name} is invalid:
{msg}")),
        },
        Err(msg) => compile_error(msg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locates_sat_dir_upwards() {
        let root = std::env::temp_dir().join(format!("sattest{}", std::process::id()));
        let sat_dir = root.join("sat");
        std::fs::create_dir_all(&sat_dir).unwrap();
        std::fs::write(
            sat_dir.join("k.sat"),
            "kernel k [workgroup(1,1,1)] (a: buf<f32>) {}",
        )
        .unwrap();
        let call = root.join("src/bin/main.rs");
        assert_eq!(sat_path(&call, "k.sat").unwrap(), sat_dir.join("k.sat"));
        assert!(sat_path(&call, "missing.sat").is_err());
        assert!(sat_path(&call, "a/b.sat").is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn validates_good_kernel() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/sat/scale.sat");
        precompile(&path).expect("scale.sat must precompile");
    }

    #[test]
    fn rejects_bad_kernel() {
        let dir = std::env::temp_dir().join(format!("satbad{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.sat");
        std::fs::write(
            &path,
            "kernel k [workgroup(1,1,1)] (a: buf<f32>) {\n    var x = 1;\n    a[0] = x;\n}",
        )
        .unwrap();
        let msg = precompile(&path).expect_err("bad kernel must fail");
        assert!(msg.contains("type mismatch"), "got: {msg}");
        assert!(msg.contains("bad.sat:3"), "got: {msg}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

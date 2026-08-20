pub fn gguf_key(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("token_embd.weight") {
        return Some(format!("embed_tokens.weight{rest}"));
    }
    if name == "output.weight" {
        return Some("lm_head.weight".into());
    }
    if name == "output_norm.weight" {
        return Some("norm.weight".into());
    }

    match name {
        "per_layer_token_embd.weight" => return Some("embed_tokens_per_layer.weight".into()),
        "per_layer_model_proj.weight" => return Some("per_layer_model_projection.weight".into()),
        "per_layer_proj_norm.weight" => return Some("per_layer_projection_norm.weight".into()),
        "rope_freqs.weight" => return None,
        _ => {}
    }
    let rest = name.strip_prefix("blk.")?;
    let (idx, tail) = rest.split_once('.')?;
    let layer: u32 = idx.parse().ok()?;

    if tail == "layer_output_scale.weight" {
        return Some(format!("layers.{layer}.layer_scalar"));
    }
    let (stem, suffix) = tail.rsplit_once('.')?;
    let canon = match stem {
        "attn_norm" => "input_layernorm",
        "attn_q" => "self_attn.q_proj",
        "attn_k" => "self_attn.k_proj",
        "attn_v" => "self_attn.v_proj",
        "attn_output" => "self_attn.o_proj",
        "attn_q_norm" => "self_attn.q_norm",
        "attn_k_norm" => "self_attn.k_norm",
        "ffn_norm" => "post_attention_layernorm",
        "ffn_gate" => "mlp.gate_proj",
        "ffn_up" => "mlp.up_proj",
        "ffn_down" => "mlp.down_proj",

        "post_attention_norm" => "post_attention_norm",
        "post_ffw_norm" => "post_ffw_norm",

        "inp_gate" => "per_layer_input_gate",
        "proj" => "per_layer_projection",
        "post_norm" => "post_per_layer_input_norm",
        "layer_output_scale" => "layer_scalar",
        _ => return None,
    };
    Some(format!("layers.{layer}.{canon}.{suffix}"))
}

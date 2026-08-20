use std::time::Instant;

use flint_error::{Error, Result};
use serde_json::json;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use crate::hub::Hub;
use crate::protocols;

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
}

pub fn serve(cfg: ServerConfig, hub: Hub) -> Result<()> {
    let addr = (cfg.host.as_str(), cfg.port);
    let server = Server::http(addr)
        .map_err(|e| Error::Model(format!("bind {}:{}: {e}", cfg.host, cfg.port)))?;
    eprintln!(
        "[server] listening on http://{}:{} (model: {})",
        cfg.host,
        cfg.port,
        hub.model_id()
    );
    for request in server.incoming_requests() {
        let hub = hub.clone();
        let api_key = cfg.api_key.clone();
        std::thread::spawn(move || {
            let t0 = Instant::now();
            let method = request.method().clone();
            let url = request.url().to_string();
            let status = route(request, &hub, api_key.as_deref());
            eprintln!(
                "[server] {} {} -> {status} ({:.1} ms)",
                method,
                url,
                t0.elapsed().as_secs_f64() * 1000.0
            );
        });
    }
    Ok(())
}

pub fn sse_headers() -> Vec<Header> {
    vec![
        Header::from_bytes(b"Content-Type", b"text/event-stream").expect("valid header"),
        Header::from_bytes(b"Cache-Control", b"no-cache").expect("valid header"),
        Header::from_bytes(b"Access-Control-Allow-Origin", b"*").expect("valid header"),
    ]
}

fn json_response(value: serde_json::Value) -> Response<std::io::Cursor<Vec<u8>>> {
    crate::protocols::json_response(value)
}

fn route(request: Request, hub: &Hub, api_key: Option<&str>) -> u16 {
    let method = request.method().clone();
    let path = request
        .url()
        .split('?')
        .next()
        .unwrap_or_default()
        .to_string();
    if method == Method::Options {
        let response = Response::empty(StatusCode(204))
            .with_header(
                Header::from_bytes(b"Access-Control-Allow-Origin", b"*").expect("valid header"),
            )
            .with_header(
                Header::from_bytes(b"Access-Control-Allow-Headers", b"*").expect("valid header"),
            )
            .with_header(
                Header::from_bytes(b"Access-Control-Allow-Methods", b"*").expect("valid header"),
            );
        let _ = request.respond(response);
        return 204;
    }
    if !authorized(&request, api_key) {
        let _ = request.respond(
            json_response(
                json!({"error": {"message": "invalid api key", "type": "authentication_error"}}),
            )
            .with_status_code(StatusCode(401)),
        );
        return 401;
    }
    match (method, path.as_str()) {
        (Method::Get, "/v1/models") => {
            let anthropic = request
                .headers()
                .iter()
                .any(|h| h.field.equiv("x-api-key") || h.field.equiv("anthropic-version"));
            let _ = request.respond(json_response(if anthropic {
                anthropic_models_json(hub)
            } else {
                models_json(hub)
            }));
            200
        }
        (Method::Get, "/healthz") | (Method::Get, "/health") => {
            let _ = request.respond(json_response(json!({"status": "ok"})));
            200
        }
        (Method::Post, "/v1/chat/completions") => {
            respond(request, |r| protocols::openai_chat::handle(r, hub))
        }
        (Method::Post, "/v1/responses") => {
            respond(request, |r| protocols::openai_responses::handle(r, hub))
        }
        (Method::Post, "/v1/messages") => {
            respond(request, |r| protocols::anthropic::handle(r, hub))
        }
        (Method::Post, "/v1/messages/count_tokens") => respond(request, |r| {
            protocols::anthropic::handle_count_tokens(r, hub)
        }),
        (Method::Post, p) if p.ends_with(":generateContent") => {
            respond(request, |r| protocols::gemini::handle(r, hub, false))
        }
        (Method::Post, p) if p.ends_with(":streamGenerateContent") => {
            respond(request, |r| protocols::gemini::handle(r, hub, true))
        }
        _ => {
            let _ = request.respond(
                json_response(json!({"error": {"message": format!("not found: {path}"), "type": "not_found"}}))
                    .with_status_code(StatusCode(404)),
            );
            404
        }
    }
}

fn respond(request: Request, f: impl FnOnce(Request) -> Result<()>) -> u16 {
    match f(request) {
        Ok(()) => 200,
        Err(e) => {
            eprintln!("[server] request failed: {e}");
            500
        }
    }
}

fn authorized(request: &Request, api_key: Option<&str>) -> bool {
    let Some(key) = api_key else {
        return true;
    };
    let bearer = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Authorization"))
        .and_then(|h| h.value.as_str().strip_prefix("Bearer "));
    let x_api = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("x-api-key"))
        .map(|h| h.value.as_str());
    let google = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("x-goog-api-key"))
        .map(|h| h.value.as_str());
    bearer == Some(key) || x_api == Some(key) || google == Some(key)
}

fn models_json(hub: &Hub) -> serde_json::Value {
    json!({
        "object": "list",
        "data": [{
            "id": hub.model_id(),
            "object": "model",
            "created": crate::protocols::now_secs(),
            "owned_by": "flint",
            "context_length": hub.context_len(),
        }]
    })
}

fn anthropic_models_json(hub: &Hub) -> serde_json::Value {
    let id = hub.model_id();
    json!({
        "data": [{
            "type": "model",
            "id": id,
            "display_name": id,
            "created_at": crate::protocols::now_secs().to_string(),
        }],
        "has_more": false,
        "first_id": id,
        "last_id": id,
    })
}

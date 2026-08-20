use axum::Router;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use flint_error::{Error, Result};
use serde_json::{Value, json};
use tower_http::cors::CorsLayer;

use crate::generator::Generator;
use crate::protocols;

#[derive(Clone)]
pub struct AppState {
    pub(crate) generator: Generator,
    api_key: Option<String>,
}

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
}

pub fn serve(cfg: ServerConfig, generator: Generator) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| Error::Model(format!("build the tokio runtime: {e}")))?;
    runtime.block_on(serve_async(cfg, generator))
}

async fn serve_async(cfg: ServerConfig, generator: Generator) -> Result<()> {
    let model_id = generator.model_id().to_string();
    let state = AppState {
        generator,
        api_key: cfg.api_key,
    };
    let app = Router::new()
        .route("/v1/models", get(models))
        .route("/healthz", get(health))
        .route("/health", get(health))
        .route("/v1/chat/completions", post(protocols::openai_chat::handle))
        .route("/v1/responses", post(protocols::openai_responses::handle))
        .route("/v1/messages", post(protocols::anthropic::handle))
        .route(
            "/v1/messages/count_tokens",
            post(protocols::anthropic::handle_count_tokens),
        )
        .route("/{*path}", post(gemini))
        .layer(middleware::from_fn_with_state(state.clone(), authorize))
        .layer(CorsLayer::very_permissive())
        .with_state(state);
    let listener = tokio::net::TcpListener::bind((cfg.host.as_str(), cfg.port))
        .await
        .map_err(|e| Error::Model(format!("bind {}:{}: {e}", cfg.host, cfg.port)))?;
    eprintln!(
        "[server] listening on http://{}:{} (model: {model_id})",
        cfg.host, cfg.port
    );
    axum::serve(listener, app)
        .await
        .map_err(|e| Error::Model(format!("serve: {e}")))
}

async fn authorize(State(state): State<AppState>, request: axum::http::Request<axum::body::Body>, next: Next) -> Response {
    if authorized(request.headers(), state.api_key.as_deref()) {
        return next.run(request).await;
    }
    (
        StatusCode::UNAUTHORIZED,
        protocols::json_response(
            json!({"error": {"message": "invalid api key", "type": "authentication_error"}}),
        ),
    )
        .into_response()
}

fn authorized(headers: &HeaderMap, api_key: Option<&str>) -> bool {
    let Some(key) = api_key else {
        return true;
    };
    let bearer = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let x_api = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok());
    let google = headers
        .get("x-goog-api-key")
        .and_then(|v| v.to_str().ok());
    bearer == Some(key) || x_api == Some(key) || google == Some(key)
}

async fn models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let anthropic = headers.contains_key("x-api-key") || headers.contains_key("anthropic-version");
    let body = if anthropic {
        anthropic_models_json(&state.generator)
    } else {
        models_json(&state.generator)
    };
    protocols::json_response(body).into_response()
}

async fn health() -> Response {
    protocols::json_response(json!({"status": "ok"})).into_response()
}

async fn gemini(State(state): State<AppState>, Path(path): Path<String>, body: Bytes) -> Response {
    let stream = if path.ends_with(":streamGenerateContent") {
        true
    } else if path.ends_with(":generateContent") {
        false
    } else {
        return (
            StatusCode::NOT_FOUND,
            protocols::json_response(
                json!({"error": {"message": format!("not found: {path}"), "type": "not_found"}}),
            ),
        )
            .into_response();
    };
    protocols::gemini::handle(State(state), body, stream).await
}

fn models_json(generator: &Generator) -> Value {
    json!({
        "object": "list",
        "data": [{
            "id": generator.model_id(),
            "object": "model",
            "created": protocols::now_secs(),
            "owned_by": "flint",
            "context_length": generator.context_len(),
        }]
    })
}

fn anthropic_models_json(generator: &Generator) -> Value {
    let id = generator.model_id();
    json!({
        "data": [{
            "type": "model",
            "id": id,
            "display_name": id,
            "created_at": protocols::now_secs().to_string(),
        }],
        "has_more": false,
        "first_id": id,
        "last_id": id,
    })
}

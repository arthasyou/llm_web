use crate::llm::generation::LogitsProcessor;
use crate::llm::{predict, LoraLLM, LLM};
use axum::{
    body::Body,
    extract::{FromRequest, Path, Query, Request},
    http::{header, HeaderMap, StatusCode},
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post},
    Extension, Json, RequestExt, Router,
};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use validator::Validate;

pub fn api() -> Router {
    Router::new()
        .route("/chat", post(chat))
        .route("/chat_lora", post(chat_lora))
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Chat {
    text: String,
}

pub async fn chat(
    Extension(mut llm): Extension<LLM>,
    Json(parames): Json<Chat>,
) -> Result<Json<Chat>, StatusCode> {
    let text = predict::run(&parames.text, &mut llm);
    let chat = Chat { text };
    Ok(Json(chat))
}

pub async fn chat_lora(
    Extension(mut llm): Extension<LoraLLM>,
    Json(parames): Json<Chat>,
) -> Result<Json<Chat>, StatusCode> {
    let text = predict::run_lora(&parames.text, &mut llm);
    let chat = Chat { text };
    Ok(Json(chat))
}

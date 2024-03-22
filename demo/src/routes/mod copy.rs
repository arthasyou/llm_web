mod routes_api;

use super::middleware::cors::create_cors;
use crate::llm::{LoraLLM, LLM};
use axum::{Extension, Router};

pub fn create_routes(llm: LoraLLM) -> Router {
    let cors = create_cors();
    let api = routes_api::api();
    // let a = Extension(llm);
    Router::new()
        .nest("/api", api)
        .layer(Extension(llm))
        .layer(cors)
}

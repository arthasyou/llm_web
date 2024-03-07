mod routes_api;

use super::middleware::cors::create_cors;
use crate::llm::LLM;
use axum::{Extension, Router};

pub fn create_routes(llm: LLM) -> Router {
    let cors = create_cors();
    let api = routes_api::api();
    Router::new()
        .nest("/api", api)
        .layer(Extension(llm))
        .layer(cors)
}

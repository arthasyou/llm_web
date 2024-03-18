mod routes_api;

use super::middleware::cors::create_cors;
use crate::llm::{LoraLLM, LLM};
// use axum::{Extension, Router};

// pub fn create_routes(llm: LoraLLM) -> Router {
//     let cors = create_cors();
//     let api = routes_api::api();
//     let a = Extension(llm);
//     Router::new()
//         .nest("/api", api)
//         .layer(Extension(llm))
//         .layer(cors)
// }

use crate::llm::predict;
use crate::SHARED_DATA;
use poem::{handler, listener::TcpListener, web::Path, Route, Server};
use std::ops::DerefMut;

#[handler]
pub fn hello(Path(name): Path<String>) -> String {
    if let Some(data_mutex) = SHARED_DATA.get() {
        if let Ok(mut llm) = data_mutex.lock() {
            let text = predict::run(&name, llm.deref_mut());
            return format!("chat: {}", text);
        } else {
            format!("runtime error")
        }
    } else {
        format!("runtime error")
    }
}

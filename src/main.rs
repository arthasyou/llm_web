// mod candle_lora;
mod error;
mod llm;
mod middleware;
mod routes;
mod utility;

use crate::llm::LLM;
use once_cell::sync::OnceCell;
use poem::{get, listener::TcpListener, Route, Server};
use routes::hello;
use std::sync::{Arc, Mutex};

pub static SHARED_DATA: OnceCell<Arc<Mutex<LLM>>> = OnceCell::new();

#[tokio::main]
async fn main() {
    // let dir = "/Users/yousx/models/meta_llama";
    let dir = "/Users/yousx/models/lora";

    // let mut llm = llm::LLM::new(dir);

    // let mut llm = llm::LoraLLM::new(dir);
    SHARED_DATA.get_or_init(|| Arc::new(Mutex::new(LLM::new(dir))));
    let app = Route::new().at("/:name", get(hello));
    _ = Server::new(TcpListener::bind("127.0.0.1:3000"))
        .run(app)
        .await
}

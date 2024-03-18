// mod candle_lora;
mod error;
mod llm;
mod middleware;
mod routes;
mod utility;

#[tokio::main]
async fn main() {
    // let dir = "/Users/yousx/models/meta_llama";
    let dir = "/Users/yousx/models/lora";

    // let mut llm = llm::LLM::new(dir);

    let mut llm = llm::LoraLLM::new(dir);

    let routes = routes::create_routes(llm);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, routes).await.unwrap();
}
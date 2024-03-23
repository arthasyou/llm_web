// mod candle_lora;
mod error;
mod llm;
mod middleware;
mod routes;
mod settings;
mod utility;

use settings::Settings;

#[tokio::main]
async fn main() {
    let settings = Settings::new().unwrap();
    println!("{:?}", settings);

    // let dir = "/Users/yousx/models/lora";

    // let mut llm = llm::LLM::new(dir);

    let mut llm = llm::LoraLLM::new(&settings.llm.model_dir, settings.llm.use_cpu);

    let routes = routes::create_routes(llm);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, routes).await.unwrap();
}

pub mod generation;
pub mod load_llm;
pub mod models;
pub mod predict;
pub mod token_output_stream;
pub mod train;
mod util;

use candle_core::{DType, Device};
use models::llama::{Cache, Llama};
use token_output_stream::TokenOutputStream;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct LLM {
    pub device: Device,
    pub model: Llama,
    pub tokenizer: Tokenizer,
    pub token_output: TokenOutputStream,
    pub cache: Cache,
}

impl LLM {
    pub fn new(dir: &str) -> Self {
        let device = Device::Cpu;
        let (model, mut cache) = load_llm::load_model(dir).unwrap();
        let tokenizer = load_llm::load_tokenizer(dir);
        let token_output = TokenOutputStream::new(tokenizer.clone());

        Self {
            device,
            model,
            tokenizer,
            token_output,
            cache,
        }
    }
}

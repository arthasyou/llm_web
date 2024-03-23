pub mod generation;
pub mod load_llm;
pub mod models;
pub mod predict;
pub mod token_output_stream;
pub mod train;
mod util;

use candle_core::Device;
use models::{
    llama::{Cache, Llama},
    lora::llama_lora::{self, LlamaLora},
};
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
        let device = load_llm::device(true).unwrap();
        let (model, mut cache) = load_llm::load_model(dir, &device).unwrap();
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

#[derive(Clone)]
pub struct LoraLLM {
    pub device: Device,
    pub model: LlamaLora,
    pub tokenizer: Tokenizer,
    pub token_output: TokenOutputStream,
    pub cache: llama_lora::Cache,
}

impl LoraLLM {
    pub fn new(dir: &str) -> Self {
        let device = load_llm::device(true).unwrap();
        let (model, cache) = load_llm::load_lora_model(dir, &device).unwrap();
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

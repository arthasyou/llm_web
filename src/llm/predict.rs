use super::generation::LogitsProcessor;

use super::LLM;
use crate::error::{Error, Result};
use candle_core::Tensor;

pub fn run(prompt: &str, llm: &mut LLM) -> String {
    // print!("{:#?}", llm);
    let mut logits_processor = LogitsProcessor::new(1981, Some(0.7_f64), Some(0.7_f64));
    // println!("{:#?}", logits_processor);

    let mut index_pos = 0;
    let mut token_generated = 0;

    let repeat_penalty = 0;

    let mut tokens = llm
        .tokenizer
        .encode(prompt, true)
        .unwrap()
        .get_ids()
        .to_vec();

    let mut result = "".to_string();

    for index in 0..10 {
        let (context_size, context_index) = if llm.cache.use_kv_cache && index > 0 {
            (1, index_pos)
        } else {
            (tokens.len(), 0)
        };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &llm.device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let logits = llm
            .model
            .forward(&input, context_index, &mut llm.cache)
            .unwrap();
        let logits = logits.squeeze(0).unwrap();

        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits).unwrap();
        token_generated += 1;
        tokens.push(next_token);

        if let Some(t) = llm.token_output.next_token(next_token).unwrap() {
            result = format!("{},{}", result, t);
        }
    }
    result
}

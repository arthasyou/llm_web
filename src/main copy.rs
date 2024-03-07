mod error;
mod llm;
mod middleware;
mod routes;
mod utility;

use candle_core::{Device, Tensor};
use llm::generation::LogitsProcessor;

#[tokio::main]
async fn main() {
    // let routes = routes::create_routes();
    // let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    // axum::serve(listener, routes).await.unwrap();

    let dir = "/Users/yousx/models/meta_llama";

    let mut llm = llm::LLM::new(dir);

    // llm::predict::run(dir);
    // println!("{:?}", model);

    let mut logits_processor = LogitsProcessor::new(1981, None, None);
    // println!("{:#?}", logits_processor);

    let mut index_pos = 0;
    let mut token_generated = 0;
    let prompt = "您好";

    let repeat_penalty = 0;

    let mut tokens = llm
        .tokenizer
        .encode(prompt, true)
        .unwrap()
        .get_ids()
        .to_vec();

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
        // let logits = if repeat_penalty == 1. {
        //     logits
        // } else {
        //     let start_at = tokens.len().saturating_sub(args.repeat_last_n);
        //     candle_transformers::utils::apply_repeat_penalty(
        //         &logits,
        //         args.repeat_penalty,
        //         &tokens[start_at..],
        //     )?
        // };
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits).unwrap();
        token_generated += 1;
        tokens.push(next_token);

        // if Some(next_token) == eos_token_id {
        //     break;
        // }
        if let Some(t) = llm.token_output.next_token(next_token).unwrap() {
            print!("{t}");
            // std::io::stdout().flush()?;
        }
    }
    // if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
    //     print!("{rest}");
    // }
    // let dt = start_gen.elapsed();
    // println!(
    //     "\n\n{} tokens generated ({} token/s)\n",
    //     token_generated,
    //     // token_generated as f64 / dt.as_secs_f64(),
    // );
}

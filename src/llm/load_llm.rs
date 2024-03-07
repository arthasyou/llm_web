use super::models::llama::{Cache, Config, Llama};
use super::predict;
use super::util::{self, Batch};
use crate::error::Result;
use candle_core::{DType, Device};
use candle_nn::VarMap;
use tokenizers::Tokenizer;

pub fn load_model(dir: &str) -> Result<(Llama, Cache)> {
    let device = Device::Cpu;
    // let device = Device::new_cuda(0)?;
    // println!("{:?}", &device);

    // ================================================================
    // initialize model
    // ================================================================

    println!("initializing model........");

    let config = Config::config_1b(false);
    let cache = Cache::new(false, DType::F32, &config, &device)?;
    let varmap = VarMap::new();
    let paths = crate::utility::find_files_with_extension(dir, "safetensors").unwrap();

    let vb = util::from_mmaped_safetensors(&varmap, &paths, DType::F32, &device, false).unwrap();
    // let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = Llama::load(vb, &config).unwrap();

    Ok((model, cache))
}

pub fn load_tokenizer(dir: &str) -> Tokenizer {
    let file = format!("{}/tokenizer.json", dir);
    println!("{}", file);
    let tokenizer = Tokenizer::from_file(&file).unwrap();
    tokenizer
}

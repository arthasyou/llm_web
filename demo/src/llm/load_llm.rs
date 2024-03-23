use super::models::{
    llama::{Cache, Config, Llama},
    lora::llama_lora::{self, LlamaLora},
};

use super::util::{self};
use crate::error::Result;
use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    DType, Device,
};
use candle_lora::{LoraConfig, LoraEmbeddingConfig, LoraLinearConfig};
use candle_nn::VarMap;
use tokenizers::Tokenizer;

pub fn load_model(dir: &str, device: &Device) -> Result<(Llama, Cache)> {
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

pub fn load_lora_model(dir: &str, device: &Device) -> Result<(LlamaLora, llama_lora::Cache)> {
    println!("initializing model........");

    let config = llama_lora::Config::config_1b(false);
    let cache = llama_lora::Cache::new(false, DType::F32, &config, &device)?;
    let varmap = VarMap::new();
    let paths = crate::utility::find_files_with_extension(dir, "safetensors").unwrap();

    let vb = util::from_mmaped_safetensors(&varmap, &paths, DType::F32, &device, false).unwrap();
    // let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let loraconfig = LoraConfig::new(1, 1., None);
    let linearconfig = LoraLinearConfig::new(config.hidden_size, config.vocab_size);
    let embedconfig = LoraEmbeddingConfig::new(config.vocab_size, config.hidden_size);

    let model = LlamaLora::load(vb, &config, false, loraconfig, linearconfig, embedconfig).unwrap();

    Ok((model, cache))
}

pub fn load_tokenizer(dir: &str) -> Tokenizer {
    let file = format!("{}/tokenizer.json", dir);
    println!("{}", file);
    let tokenizer = Tokenizer::from_file(&file).unwrap();
    tokenizer
}

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

use crate::error::{Error, Result};
use candle_core::{DType, Device, Var};
use candle_core::{IndexOp, Tensor};
use candle_nn::var_builder::{SimpleBackend, VarBuilderArgs};
use candle_nn::{VarBuilder, VarMap};
use rand::thread_rng;
use rand::Rng;
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::Read,
    path::Path,
};
use tqdm::Iter;
use walkdir::WalkDir;

pub fn sorted_char(text: &str) -> Vec<char> {
    let mut chars: Vec<char> = HashSet::<char>::from_iter(text.chars())
        .into_iter()
        .collect();
    chars.sort_unstable();
    chars
}

pub fn tokenization(chars: &Vec<char>) -> (HashMap<char, u32>, HashMap<u32, char>) {
    let string_to_int: HashMap<char, u32> = chars
        .iter()
        .enumerate()
        .map(|(i, &ch)| (ch, i as u32))
        .collect();
    let int_to_string: HashMap<u32, char> = chars
        .iter()
        .enumerate()
        .map(|(i, &ch)| (i as u32, ch))
        .collect();
    (string_to_int, int_to_string)
}

pub fn encode(s: &str, mapping: &HashMap<char, u32>) -> Vec<u32> {
    s.chars().filter_map(|c| mapping.get(&c)).cloned().collect()
}

pub fn decode(code: &[u32], mapping: &HashMap<u32, char>) -> String {
    code.iter()
        .filter_map(|&i| mapping.get(&i))
        .cloned()
        .collect()
}

pub fn split_data(data: Vec<u32>) -> (Vec<u32>, Vec<u32>) {
    let len = data.len();

    let n = (0.8 * len as f32) as usize;
    let train_data = &data[..n];
    let val_data = &data[n..];
    let train_data: Vec<u32> = train_data.iter().map(|&x| x).collect();
    let val_data: Vec<u32> = val_data.iter().map(|&x| x).collect();
    (train_data, val_data)
}

#[derive(Debug)]
pub struct Block {
    pub x: Vec<Vec<u32>>,
    pub y: Vec<Vec<u32>>,
    pub size: usize,
}

#[derive(Debug)]
pub struct Batch {
    pub x: Tensor,
    pub y: Tensor,
}

impl Batch {
    pub fn get_batch(data: &Tensor, block_size: usize, batch_size: usize) -> Self {
        let mut rng = thread_rng();
        let len = data.dims1().unwrap();
        let random_numbers: Vec<usize> = (0..batch_size)
            .map(|_| rng.gen_range(0..(len - block_size)))
            .collect();

        let x: Vec<Tensor> = random_numbers
            .iter()
            .map(|&i| data.i(i..i + block_size).unwrap())
            .collect();
        let x = Tensor::stack(&x, 0).unwrap();

        let y: Vec<Tensor> = random_numbers
            .iter()
            .map(|&i| data.i(i + 1..i + block_size + 1).unwrap())
            .collect();
        let y = Tensor::stack(&y, 0).unwrap();

        Self { x, y }
    }
}

// 从一组Vec<f32>随机抽取几个样本
pub fn multinomial(probs: &[f32], num_samples: usize) -> Vec<u32> {
    let mut rng = rand::thread_rng();
    let mut samples: Vec<u32> = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let random_value: f32 = rng.gen(); // 生成随机值 [0, 1)
        let mut cumulative_prob = 0.0;

        for (idx, &prob) in probs.iter().enumerate() {
            cumulative_prob += prob;

            if random_value < cumulative_prob {
                samples.push(idx as u32);
                break;
            }
        }
    }

    samples
}

pub fn load_txt_files<P: AsRef<Path>>(path: P) -> Result<String> {
    let mut contents = String::new();

    for entry in WalkDir::new(path) {
        let entry = entry.map_err(|e| Error::Norm {
            message: e.to_string(),
        })?;
        let path = entry.path();

        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt") {
            let mut file = fs::File::open(path).map_err(|e| Error::Norm {
                message: e.to_string(),
            })?;
            let mut text = String::new();
            file.read_to_string(&mut text).map_err(|e| Error::Norm {
                message: e.to_string(),
            })?;
            contents += &text;
        }
    }

    Ok(contents)
}

/// Load tensors into a VarBuilder backed by a VarMap using MmapedSafetensors.
/// Set `silent` to not show a progress bar.
pub fn from_mmaped_safetensors<'a, P: AsRef<Path>>(
    map: &VarMap,
    paths: &[P],
    dtype: DType,
    device: &Device,
    silent: bool,
) -> Result<VarBuilderArgs<'a, Box<dyn SimpleBackend>>> {
    {
        let mut ws = map.data().lock().unwrap();

        let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::multi(paths)? };

        if silent {
            for (name, _) in tensors.tensors() {
                let tensor = tensors
                    .load(&name, device)?
                    .to_device(device)?
                    .to_dtype(dtype)?;
                ws.insert(name.clone(), Var::from_tensor(&tensor)?);
            }
        } else {
            for (name, _) in tensors.tensors().iter().tqdm() {
                let tensor = tensors
                    .load(name, device)?
                    .to_device(device)?
                    .to_dtype(dtype)?;
                ws.insert(name.clone(), Var::from_tensor(&tensor)?);
            }
        };
    }

    Ok(VarBuilder::from_varmap(&map, dtype, device))
}

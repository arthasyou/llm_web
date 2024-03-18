use std::{
    collections::HashMap,
    f32::consts::E,
    sync::{Arc, Mutex},
};

use crate::error::{Error, Result};

use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{embedding, linear, Embedding, Linear, RmsNorm, VarBuilder};

// pub const MAX_SEQ_LEN: usize = 4096;
pub const MAX_SEQ_LEN: usize = 128;

#[derive(Debug)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
}

impl Config {
    pub fn config_1b(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 256,
            intermediate_size: 512,
            vocab_size: 32000,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            use_flash_attn,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        }
    }

    pub fn config_7b(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Cache {
    masks: Arc<Mutex<HashMap<usize, Tensor>>>,
    pub use_kv_cache: bool,
    // #[allow(clippy::type_complexity)]
    kvs: Arc<Mutex<Vec<Option<(Tensor, Tensor)>>>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl Cache {
    pub fn new(use_kv_cache: bool, cfg: &Config, dtype: DType, device: &Device) -> Result<Self> {
        // 预先计算Positional Encoding不同的频率
        let n_elem = cfg.hidden_size / cfg.num_attention_heads;
        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / n_elem as f32))
            .collect();

        let theta = Tensor::new(theta.as_slice(), device)?;

        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;

        let idx_theta = Tensor::cat(&[&idx_theta, &idx_theta], D::Minus1).unwrap();
        let cos = idx_theta.cos().unwrap().to_dtype(dtype).unwrap();
        let sin = idx_theta.sin().unwrap().to_dtype(dtype).unwrap();

        Ok(Self {
            device: device.clone(),
            cos,
            sin,
            use_kv_cache,
            masks: Arc::new(Mutex::new(HashMap::new())),
            kvs: Arc::new(Mutex::new(vec![None; cfg.num_hidden_layers])),
        })
    }

    fn mask(&self, t: usize) -> Result<Tensor> {
        let mut masks = self.masks.lock().unwrap();
        if let Some(mask) = masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}

#[derive(Debug)]
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    cache: Cache,
    use_flash_attn: bool,
}

impl CausalSelfAttention {
    fn init(vb: VarBuilder, cache: &Cache, cfg: &Config) -> Result<Self> {
        let size_in = cfg.hidden_size;

        //下面2行代码看起来好像是冗余，实际上为了严格对齐于hidden_size. 有可能hidden_size不是attention heads的整数倍
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;

        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            use_flash_attn: cfg.use_flash_attn,
            cache: cache.clone(),
        })
    }

    fn forward(&self, x: &Tensor, index_pos: usize, block_idx: usize) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let mut k = self.apply_rotary_emb(&k, index_pos)?;

        if self.cache.use_kv_cache {
            let mut cache = self.cache.kvs.lock().unwrap();
            if let Some((cache_k, cache_v)) = &cache[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
                let k_seq_len = k.dims()[1];
                if k_seq_len > MAX_SEQ_LEN {
                    k = k
                        .narrow(D::Minus1, k_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                        .contiguous()?
                }
                let v_seq_len = v.dims()[1];
                if v_seq_len > 2 * MAX_SEQ_LEN {
                    v = v
                        .narrow(D::Minus1, v_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                        .contiguous()?
                }
            }
            cache[block_idx] = Some((k.clone(), v.clone()))
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let y = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?
        } else {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let mask = self.cache.mask(seq_len)?.broadcast_as(att.shape())?;
            let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        let y = self.o_proj.forward(&y)?;

        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(2)?
                .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
                .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
            Ok(x)
        }
    }

    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, _, seq_len, hidden_size) = x.dims4()?;
        let cos = self.cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.cache.sin.narrow(0, index_pos, seq_len)?;
        let cos = cos.broadcast_as((b_sz, 1, seq_len, hidden_size))?;
        let sin = sin.broadcast_as((b_sz, 1, seq_len, hidden_size))?;
        let x1 = x.narrow(D::Minus1, 0, hidden_size / 2)?;
        let x2 = x.narrow(D::Minus1, hidden_size / 2, hidden_size / 2)?;
        let rotate_x = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        let rope = (x.broadcast_mul(&cos)? + rotate_x.broadcast_mul(&sin)?)?;
        Ok(rope)
    }
}

#[derive(Debug)]
struct Mlp {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn init(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // println!("going here line: {}", line!());
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x)?)? * self.c_fc2.forward(x)?)?;
        self.c_proj.forward(&x).map_err(|e| Error::Norm {
            message: e.to_string(),
        })
    }
}

#[derive(Debug)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn init(vb: VarBuilder, cache: &Cache, cfg: &Config) -> Result<Self> {
        let attn = CausalSelfAttention::init(vb.pp("self_attn"), cache, cfg)?;
        let mlp = Mlp::init(vb.pp("mlp"), cfg)?;
        let rms_1 =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, index_pos: usize, block_idx: usize) -> Result<Tensor> {
        // println!("going here line: {}", line!());
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }
}

#[derive(Debug)]
pub struct Llama {
    wte: Embedding, // wte (World Token embedding)
    blocks: Vec<Block>,
    lnf: RmsNorm, // lnf (Layer Norm final)
    lm_head: Linear,
}

impl Llama {
    pub fn new(vb: VarBuilder, cache: &Cache, cfg: &Config) -> Result<Self> {
        let wte = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("token"))?;
        let lm_head = candle_nn::linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let lnf = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let blocks = (0..cfg.num_hidden_layers)
            .map(|i| Block::init(vb.pp(&format!("model.layers.{i}")), cache, cfg).unwrap())
            .collect();
        Ok(Self {
            wte,
            lnf,
            lm_head,
            blocks,
        })
    }

    pub fn forward(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        // println!("going here line: {}", line!());
        // let (batch_size, seq_len) = x.dims2()?;
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx)?;
        }
        let x = self.lnf.forward(&x)?;
        let (batch_size, seq_len, c) = x.dims3()?;
        // let x = x.i((.., seq_len - 1, ..))?;
        let x = x.reshape((batch_size * seq_len, c))?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32).map_err(|e| Error::Norm {
            message: e.to_string(),
        })
    }

    // pub fn generate(&self, input: &Tensor, max_tokens: usize) -> Vec<u32> {
    //     let mut results: Vec<u32> = Vec::new();
    //     let mut iii = input.clone();
    //     for _ in 0..max_tokens {
    //         let (logits, _) = self.forward(&iii, None);
    //         let (b, t, c) = logits.dims3().unwrap();
    //         let logits = logits.reshape((b * t, c)).unwrap();
    //         let probs = softmax(&logits, 1)
    //             .unwrap()
    //             .flatten_all()
    //             .unwrap()
    //             .to_vec1::<f32>()
    //             .unwrap();

    //         let a = multinomial(&probs, 1);
    //         results.push(*a.get(0).unwrap());

    //         // iii = Tensor::from_vec(a, (1, 1), &self.device).unwrap();
    //     }
    //     results
    // }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

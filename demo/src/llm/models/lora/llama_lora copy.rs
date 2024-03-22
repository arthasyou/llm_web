//! The LLama2 model.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_lora::{EmbeddingLayerLike, LoraConfig, LoraEmbeddingConfig, LoraLinearConfig};
use candle_lora_macro::{self, replace_layer_fields, AutoLoraConvert};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// pub const MAX_SEQ_LEN: usize = 4096;
pub const MAX_SEQ_LEN: usize = 128;

#[derive(Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn into_config(self, use_flash_attn: bool) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads.unwrap_or(self.num_attention_heads),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            use_flash_attn,
        }
    }
}

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

// We wrap the `LlamaLinear` layer here to add some tracing so that it's easier to profile the resulting
// model.
#[derive(Debug, AutoLoraConvert, Clone)]
#[replace_layer_fields]
pub struct LlamaLinear {
    inner: candle_nn::Linear,
    span: tracing::Span,
}

impl Module for LlamaLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Clone)]
pub struct Cache {
    masks: Arc<Mutex<HashMap<usize, Tensor>>>,
    pub use_kv_cache: bool,
    #[allow(clippy::type_complexity)]
    kvs: Arc<Mutex<Vec<Option<(Tensor, Tensor)>>>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl Cache {
    pub fn new(use_kv_cache: bool, config: &Config, dtype: DType, device: &Device) -> Result<Self> {
        // precompute freqs_cis
        let n_elem = config.hidden_size / config.num_attention_heads;
        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / n_elem as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device).unwrap();
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .reshape((MAX_SEQ_LEN, 1))
            .unwrap()
            .matmul(&theta.reshape((1, theta.elem_count())).unwrap())
            .unwrap();
        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let idx_theta = Tensor::cat(&[&idx_theta, &idx_theta], D::Minus1).unwrap();
        let cos = idx_theta.cos().unwrap().to_dtype(dtype).unwrap();
        let sin = idx_theta.sin().unwrap().to_dtype(dtype).unwrap();
        Ok(Self {
            masks: Arc::new(Mutex::new(HashMap::new())),
            use_kv_cache,
            kvs: Arc::new(Mutex::new(vec![None; config.num_hidden_layers])),
            device: device.clone(),
            cos,
            sin,
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
            let mask = Tensor::from_slice(&mask, (t, t), &self.device).unwrap();
            masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}

fn linear(size1: usize, size2: usize, vb: VarBuilder) -> Result<LlamaLinear> {
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    let inner = candle_nn::linear_no_bias(size1, size2, vb).unwrap();
    Ok(LlamaLinear { inner: inner, span })
}

fn embedding(cfg: &Config, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((cfg.vocab_size, cfg.hidden_size), "weight").unwrap();
    Ok(Embedding::new(embeddings, cfg.hidden_size))
}

#[replace_layer_fields]
#[derive(AutoLoraConvert, Clone)]
struct RmsNorm {
    inner: candle_nn::RmsNorm,
    span: tracing::Span,
}

impl RmsNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = candle_nn::rms_norm(size, eps, vb).unwrap();
        Ok(Self { inner, span })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

#[replace_layer_fields]
#[derive(AutoLoraConvert, Clone)]
struct CausalSelfAttention {
    q_proj: LlamaLinear,
    k_proj: LlamaLinear,
    v_proj: LlamaLinear,
    o_proj: LlamaLinear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    cache: Cache,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
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

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (b_sz, _, seq_len, hidden_size) = x.dims4().unwrap();
        let cos = self.cache.cos.narrow(0, index_pos, seq_len).unwrap();
        let sin = self.cache.sin.narrow(0, index_pos, seq_len).unwrap();
        let cos = cos.broadcast_as((b_sz, 1, seq_len, hidden_size)).unwrap();
        let sin = sin.broadcast_as((b_sz, 1, seq_len, hidden_size)).unwrap();
        let x1 = x.narrow(D::Minus1, 0, hidden_size / 2).unwrap();
        let x2 = x
            .narrow(D::Minus1, hidden_size / 2, hidden_size / 2)
            .unwrap();
        let rotate_x = Tensor::cat(&[&x2.neg().unwrap(), &x1], D::Minus1).unwrap();
        let rope =
            (x.broadcast_mul(&cos).unwrap() + rotate_x.broadcast_mul(&sin).unwrap()).unwrap();
        Ok(rope)
    }

    fn forward(&self, x: &Tensor, index_pos: usize, block_idx: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, hidden_size) = x.dims3().unwrap();
        let q = self.q_proj.forward(x).unwrap();
        let k = self.k_proj.forward(x).unwrap();
        let v = self.v_proj.forward(x).unwrap();

        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let mut v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        let q = self.apply_rotary_emb(&q, index_pos).unwrap();
        let mut k = self.apply_rotary_emb(&k, index_pos).unwrap();

        if self.cache.use_kv_cache {
            let mut cache = self.cache.kvs.lock().unwrap();
            if let Some((cache_k, cache_v)) = &cache[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 2)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                v = Tensor::cat(&[cache_v, &v], 2)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                let k_seq_len = k.dims()[1];
                if k_seq_len > MAX_SEQ_LEN {
                    k = k
                        .narrow(D::Minus1, k_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)
                        .unwrap()
                        .contiguous()
                        .unwrap()
                }
                let v_seq_len = v.dims()[1];
                if v_seq_len > 2 * MAX_SEQ_LEN {
                    v = v
                        .narrow(D::Minus1, v_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)
                        .unwrap()
                        .contiguous()
                        .unwrap()
                }
            }
            cache[block_idx] = Some((k.clone(), v.clone()))
        }

        let k = self.repeat_kv(k).unwrap();
        let v = self.repeat_kv(v).unwrap();

        let y = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2).unwrap();
            let k = k.transpose(1, 2).unwrap();
            let v = v.transpose(1, 2).unwrap();
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)
                .unwrap()
                .transpose(1, 2)
                .unwrap()
        } else {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32).unwrap();
            let k = k.to_dtype(DType::F32).unwrap();
            let v = v.to_dtype(DType::F32).unwrap();
            let att = (q.matmul(&k.t().unwrap()).unwrap() / (self.head_dim as f64).sqrt()).unwrap();
            let mask = self
                .cache
                .mask(seq_len)
                .unwrap()
                .broadcast_as(att.shape())
                .unwrap();
            let att = masked_fill(&att, &mask, f32::NEG_INFINITY).unwrap();
            let att = candle_nn::ops::softmax(&att, D::Minus1).unwrap();
            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous().unwrap())
                .unwrap()
                .to_dtype(in_dtype)
                .unwrap()
        };
        let y = y
            .transpose(1, 2)
            .unwrap()
            .reshape(&[b_sz, seq_len, hidden_size])
            .unwrap();
        let y = self.o_proj.forward(&y).unwrap();
        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4().unwrap();
            let x = x
                .unsqueeze(2)
                .unwrap()
                .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))
                .unwrap()
                .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
                .unwrap();
            Ok(x)
        }
    }

    fn load(
        vb: VarBuilder,
        cache: &Cache,
        cfg: &Config,
        merge: bool,
        lora_config: LoraConfig,
        linear_config: LoraLinearConfig,
    ) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj")).unwrap();
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj")).unwrap();
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj")).unwrap();
        let o_proj = linear(size_q, size_in, vb.pp("o_proj")).unwrap();

        let mut this = Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            cache: cache.clone(),
            use_flash_attn: cfg.use_flash_attn,
            span,
            span_rot,
        };

        if merge {
            this.get_merged_lora_model(
                lora_config,
                &vb.pp("lora_llama_csa"),
                Some(linear_config),
                None,
                None,
                None,
            )
        } else {
            this.get_lora_model(
                lora_config,
                &vb.pp("lora_llama_csa"),
                Some(linear_config),
                None,
                None,
                None,
            )
        }

        Ok(this)
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())
        .unwrap()
        .broadcast_as(shape.dims())
        .unwrap();
    let m = mask.where_cond(&on_true, on_false).unwrap();
    Ok(m)
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: LlamaLinear,
    c_fc2: LlamaLinear,
    c_proj: LlamaLinear,
    span: tracing::Span,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (candle_nn::ops::silu(&self.c_fc1.forward(x).unwrap()).unwrap()
            * self.c_fc2.forward(x).unwrap())
        .unwrap();
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj")).unwrap();
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj")).unwrap();
        let c_proj = linear(i_size, h_size, vb.pp("down_proj")).unwrap();
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
            span,
        })
    }
}

#[replace_layer_fields]
#[derive(AutoLoraConvert, Clone)]
struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
    span: tracing::Span,
}

impl Block {
    fn forward(&self, x: &Tensor, index_pos: usize, block_idx: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x).unwrap();
        let x = (self.attn.forward(&x, index_pos, block_idx).unwrap() + residual).unwrap();
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x).unwrap()).unwrap() + residual).unwrap();
        Ok(x)
    }

    fn load(
        vb: VarBuilder,
        cache: &Cache,
        cfg: &Config,
        merge: bool,
        lora_config: LoraConfig,
        linear_config: LoraLinearConfig,
        embed_onfig: LoraEmbeddingConfig,
    ) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(
            vb.pp("self_attn"),
            cache,
            cfg,
            merge,
            lora_config.clone(),
            linear_config.clone(),
        )
        .unwrap();
        let mlp = Mlp::load(vb.pp("mlp"), cfg).unwrap();
        let rms_1 =
            RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm")).unwrap();
        let rms_2 = RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )
        .unwrap();

        let mut this = Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
        };

        if merge {
            this.get_merged_lora_model(
                lora_config,
                &vb.pp("lora_llama_block"),
                Some(linear_config),
                None,
                None,
                Some(embed_onfig),
            )
        } else {
            this.get_lora_model(
                lora_config,
                &vb.pp("lora_llama_block"),
                Some(linear_config),
                None,
                None,
                Some(embed_onfig),
            )
        }

        Ok(this)
    }
}

#[replace_layer_fields]
#[derive(AutoLoraConvert, Clone)]
pub struct LlamaLora {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: LlamaLinear,
}

impl LlamaLora {
    pub fn forward(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        // let (_b_sz, seq_len) = x.dims2().unwrap();
        let mut x = self.wte.forward(x).unwrap();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx).unwrap();
        }
        let x = self.ln_f.forward(&x).unwrap();
        // let x = x.i((.., seq_len - 1, ..)).unwrap();
        let (batch_size, seq_len, c) = x.dims3()?;
        let x = x.reshape((batch_size * seq_len, c))?;
        let logits = self.lm_head.forward(&x).unwrap();
        logits.to_dtype(DType::F32)
    }

    /// Load a Mistral model which will be converted to a LoRA model.
    ///
    /// The `merge` parameter merges the weights.
    pub fn load(
        vb: VarBuilder,
        cache: &Cache,
        cfg: &Config,
        merge: bool,
        lora_config: LoraConfig,
        linear_config: LoraLinearConfig,
        embed_config: LoraEmbeddingConfig,
    ) -> Result<Self> {
        let wte = embedding(cfg, vb.pp("model.embed_tokens")).unwrap();
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head")).unwrap();
        let ln_f = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm")).unwrap();
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| {
                Block::load(
                    vb.pp(&format!("model.layers.{i}")),
                    cache,
                    cfg,
                    merge,
                    lora_config.clone(),
                    linear_config.clone(),
                    embed_config.clone(),
                )
                .unwrap()
            })
            .collect();

        let mut this = Self {
            wte: Arc::new(wte),
            blocks,
            ln_f,
            lm_head,
        };

        if merge {
            this.get_merged_lora_model(
                lora_config,
                &vb.pp("lora_llama"),
                Some(linear_config),
                None,
                None,
                Some(embed_config),
            )
        } else {
            this.get_lora_model(
                lora_config,
                &vb.pp("lora_llama"),
                Some(linear_config),
                None,
                None,
                Some(embed_config),
            )
        }

        Ok(this)
    }
}

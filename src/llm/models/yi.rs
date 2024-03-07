/// https://huggingface.co/01-ai/Yi-6B/blob/main/modeling_yi.py
use super::with_tracing::{linear_no_bias, Linear};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

impl Config {
    pub fn config_1b() -> Self {
        Self {
            vocab_size: 64000,
            hidden_size: 1024,
            intermediate_size: 2048,
            num_hidden_layers: 16,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            hidden_act: Activation::Silu,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.,
        }
    }

    pub fn config_6b() -> Self {
        Self {
            vocab_size: 64000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            hidden_act: Activation::Silu,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 5_000_000.,
        }
    }

    pub fn config_34b() -> Self {
        Self {
            vocab_size: 64000,
            hidden_size: 7168,
            intermediate_size: 20480,
            num_hidden_layers: 60,
            num_attention_heads: 56,
            num_key_value_heads: 8,
            hidden_act: Activation::Silu,
            max_position_embeddings: 4096,
            rms_norm_eps: 1e-5,
            rope_theta: 5_000_000.,
        }
    }
}

#[derive(Debug, Clone)]
struct RmsNorm {
    inner: candle_nn::RmsNorm,
    span: tracing::Span,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = candle_nn::rms_norm(size, eps, vb).unwrap();
        Ok(Self { inner, span })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1).unwrap();
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2).unwrap();
    let xs2 = xs
        .narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)
        .unwrap();
    Tensor::cat(&[&xs2.neg().unwrap(), &xs1], D::Minus1)
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
            .reshape((max_seq_len, 1))
            .unwrap();
        let freqs = t.matmul(&inv_freq).unwrap();
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1).unwrap();
        Ok(Self {
            sin: freqs.sin().unwrap(),
            cos: freqs.cos().unwrap(),
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4().unwrap();
        let cos = self.cos.narrow(0, seqlen_offset, seq_len).unwrap();
        let sin = self.sin.narrow(0, seqlen_offset, seq_len).unwrap();
        let cos = cos.unsqueeze(0).unwrap().unsqueeze(0).unwrap(); // (1, 1, seq_len, dim)
        let sin = sin.unsqueeze(0).unwrap().unsqueeze(0).unwrap(); // (1, 1, seq_len, dim)
        let q_embed =
            (q.broadcast_mul(&cos).unwrap() + rotate_half(q).unwrap().broadcast_mul(&sin)).unwrap();
        let k_embed =
            (k.broadcast_mul(&cos).unwrap() + rotate_half(k).unwrap().broadcast_mul(&sin)).unwrap();
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj")).unwrap();
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj")).unwrap();
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj")).unwrap();
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs
            .apply(&self.gate_proj)
            .unwrap()
            .apply(&self.act_fn)
            .unwrap();
        let rhs = xs.apply(&self.up_proj).unwrap();
        (lhs * rhs).unwrap().apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = linear_no_bias(hidden_sz, num_heads * head_dim, vb.pp("q_proj")).unwrap();
        let k_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj")).unwrap();
        let v_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj")).unwrap();
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj")).unwrap();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn repeat_kv(&self, xs: Tensor) -> Result<Tensor> {
        let n_rep = self.num_kv_groups;
        if n_rep == 1 {
            Ok(xs)
        } else {
            let (b_sz, num_kv_heads, seq_len, head_dim) = xs.dims4().unwrap();
            xs.unsqueeze(2)
                .unwrap()
                .expand((b_sz, num_kv_heads, n_rep, seq_len, head_dim))
                .unwrap()
                .reshape((b_sz, num_kv_heads * n_rep, seq_len, head_dim))
        }
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3().unwrap();

        let query_states = self.q_proj.forward(xs).unwrap();
        let key_states = self.k_proj.forward(xs).unwrap();
        let value_states = self.v_proj.forward(xs).unwrap();

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        let (query_states, key_states) = self
            .rotary_emb
            .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)
            .unwrap();

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2).unwrap();
                let value_states = Tensor::cat(&[prev_v, &value_states], 2).unwrap();
                (key_states, value_states)
            }
        };
        // self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = self.repeat_kv(key_states).unwrap();
        let value_states = self.repeat_kv(value_states).unwrap();

        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states
                .matmul(&key_states.transpose(2, 3).unwrap())
                .unwrap()
                * scale)
                .unwrap();

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask).unwrap(),
            };

            // println!("attn_weights: {:?}", attn_weights);
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights).unwrap();

            attn_weights.matmul(&value_states).unwrap()
        };
        attn_output
            .transpose(1, 2)
            .unwrap()
            .reshape((b_sz, q_len, self.hidden_size))
            .unwrap()
            .apply(&self.o_proj)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn")).unwrap();
        let mlp = MLP::new(cfg, vb.pp("mlp")).unwrap();
        let ln1 =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm")).unwrap();
        let ln2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )
        .unwrap();
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.ln1.forward(xs).unwrap();
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, seqlen_offset)
            .unwrap();
        let xs = (xs + residual).unwrap();
        let residual = &xs;
        let xs = xs.apply(&self.ln2).unwrap().apply(&self.mlp).unwrap();
        residual + xs
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device();
        // println!("device: {device:.unwrap()}");
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens")).unwrap();
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, device).unwrap());
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx)).unwrap();
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm")).unwrap();
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head")).unwrap();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        // Sliding window mask.unwrap()
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device).unwrap();
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device).unwrap();
            Tensor::cat(&[&mask0, &mask], D::Minus1).unwrap()
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))
            .unwrap()
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2().unwrap();
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self
                .prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset)
                .unwrap();
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(input_ids).unwrap();

        for layer in self.layers.iter_mut() {
            xs = layer
                .forward(&xs, attention_mask.as_ref(), seqlen_offset)
                .unwrap()
        }

        let xs = xs.apply(&self.norm).unwrap().apply(&self.lm_head).unwrap();
        let (b, t, c) = xs.dims3().unwrap();
        xs.reshape((b * t, c))
        // xs.narrow(1, seq_len - 1, 1).unwrap()
        //     .apply(&self.norm).unwrap()
        //     .apply(&self.lm_head)
    }
}

"""
LogLLM: encoder-agnostic (BERT / RoBERTa / DeBERTa-v3 / Longformer / BigBird)
+ Llama 3.x QLoRA + LayerAggregator (last-K, masked pooling) + time fusion.
"""
import os
import math
import typing

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer, AutoConfig, AutoModel, AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType,
)


# ============================================================
# Encoder presets
# ============================================================

ENCODER_PRESETS: typing.Dict[str, dict] = {
    "bert-base":            {"path": "bert-base-uncased",              "attention_window": None},
    "bert-large":           {"path": "bert-large-uncased",             "attention_window": None},
    "roberta-base":         {"path": "roberta-base",                   "attention_window": None},
    "roberta-large":        {"path": "roberta-large",                  "attention_window": None},
    "distilroberta-base":   {"path": "distilroberta-base",             "attention_window": None},
    "deberta-v3-small":     {"path": "microsoft/deberta-v3-small",     "attention_window": None},
    "deberta-v3-base":      {"path": "microsoft/deberta-v3-base",      "attention_window": None},
    "deberta-v3-large":     {"path": "microsoft/deberta-v3-large",     "attention_window": None},
    "longformer-base":      {"path": "allenai/longformer-base-4096",   "attention_window": 512},
    "longformer-large":     {"path": "allenai/longformer-large-4096",  "attention_window": 512},
    "bigbird-roberta-base": {"path": "google/bigbird-roberta-base",    "attention_window": 512},
    "bigbird-roberta-large":{"path": "google/bigbird-roberta-large",   "attention_window": 512},
}


def resolve_encoder(encoder: str, overrides: typing.Optional[dict] = None) -> dict:
    """Resolve preset name OR hf path → spec dict."""
    spec = dict(ENCODER_PRESETS.get(encoder, {"path": encoder,
                                              "attention_window": None}))
    if overrides:
        spec.update(overrides)
    return spec


def _auto_lora_targets(model) -> typing.List[str]:
    names = {n for n, _ in model.named_modules()}
    for cand in (("query", "value"),                # BERT/RoBERTa/Longformer/BigBird
                 ("query_proj", "value_proj"),      # DeBERTa-v2/v3
                 ("q_lin", "v_lin")):
        if all(any(n.endswith(c) for n in names) for c in cand):
            return list(cand)
    print("[encoder] LoRA targets not auto-detected, fallback=('query','value')")
    return ["query", "value"]


def _is_longformer_like(cfg) -> bool:
    return hasattr(cfg, "attention_window")


_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16,
           "fp32": torch.float32, "tf32": torch.float32}


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"unknown dtype={name!r}; expected one of {list(_DTYPES)}")
    if name == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return _DTYPES[name]


# ============================================================
# Time features
# ============================================================

class Time2Vec(nn.Module):
    def __init__(self, k: int = 16):
        super().__init__()
        self.k = k
        self.w0 = nn.Parameter(torch.randn(1) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(k) * 0.01)
        self.b = nn.Parameter(torch.zeros(k))

    def forward(self, t):
        if t.dim() == 2:
            t = t.squeeze(-1)
        lin = (self.w0 * t + self.b0).unsqueeze(-1)
        per = torch.sin(t.unsqueeze(-1) * self.w + self.b)
        return torch.cat([lin, per], dim=-1)


def _compute_deltas(times):
    out = []
    for i, t in enumerate(times):
        if i == 0 or t is None or times[i - 1] is None:
            out.append(0.0)
        else:
            out.append(float((t - times[i - 1]).total_seconds()))
    return torch.tensor(out, dtype=torch.float32)


def _cyclic_feats(times):
    out = []
    for t in times:
        if t is None:
            out.append([0.0] * 10); continue
        h, m = t.hour / 24.0, t.minute / 60.0
        dow, mon, dom = t.weekday() / 7.0, (t.month - 1) / 12.0, (t.day - 1) / 31.0
        out.append([
            math.sin(2*math.pi*h), math.cos(2*math.pi*h),
            math.sin(2*math.pi*m), math.cos(2*math.pi*m),
            math.sin(2*math.pi*dow), math.cos(2*math.pi*dow),
            math.sin(2*math.pi*mon), math.cos(2*math.pi*mon),
            math.sin(2*math.pi*dom), math.cos(2*math.pi*dom),
        ])
    return torch.tensor(out, dtype=torch.float32)


class TimeFeatureEncoder(nn.Module):
    def __init__(self, out_dim: int, emb_len: int = 16,
                 use_cyclic: bool = True, use_delta: bool = True,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.use_delta, self.use_cyclic = use_delta, use_cyclic
        self.out_dtype = dtype
        self.t2v = Time2Vec(emb_len) if use_delta else None
        in_dim = ((emb_len + 1) if use_delta else 0) + (10 if use_cyclic else 0)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim > 0 else None

    def _in_dim(self):
        return ((self.t2v.k + 1) if self.use_delta else 0) \
               + (10 if self.use_cyclic else 0)

    def forward(self, times_list):
        B = len(times_list)
        Ls = [len(t) for t in times_list]
        L_max = max(Ls) if Ls else 1
        params = list(self.parameters())
        device = params[0].device if params else torch.device("cpu")
        feats = torch.zeros(B, L_max, self._in_dim(), device=device)   # fp32
        for b, t_list in enumerate(times_list):
            L = len(t_list); parts = []
            if self.use_delta:
                parts.append(self.t2v(_compute_deltas(t_list).to(device)))
            if self.use_cyclic:
                parts.append(_cyclic_feats(t_list).to(device))
            if parts:
                feats[b, :L] = torch.cat(parts, dim=-1)
        if self.proj is not None:
            feats = self.proj(feats)
        return feats


class CrossAttentionFusion(nn.Module):
    """Time queries → log keys/values. No causal mask (classification)."""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                           batch_first=True)
        self.norm = nn.LayerNorm(dim)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, time_emb, log_emb, log_mask=None):
        kpm = (log_mask == 0) if log_mask is not None else None
        out, _ = self.attn(time_emb, log_emb, log_emb, key_padding_mask=kpm)
        return self.norm(time_emb + out)


# ============================================================
# LayerAggregator — как в LogAnomalyModel
# ============================================================

class LayerAggregator(nn.Module):
    def __init__(self, hidden: int, k: int = 1, mode: str = "weighted",
                 pooling: str = "mean"):
        super().__init__()
        self.k = k
        self.mode = mode
        self.pooling = pooling
        if mode == "weighted":
            self.layer_weights = nn.Parameter(torch.zeros(k))
        elif mode == "attn_pool":
            self.pool_q = nn.Parameter(torch.randn(hidden) * 0.02)
        elif mode not in ("last", "mean"):
            raise ValueError(f"unknown layer agg mode={mode!r}")
        if pooling not in ("cls", "mean", "max"):
            raise ValueError(f"unknown pooling={pooling!r}")

    def forward(self, hidden_states, attention_mask):
        hs = torch.stack(hidden_states[-self.k:], dim=0)   # [K,B,L,D]

        if self.mode == "attn_pool":
            h = hs.mean(dim=0)
            score = (h @ self.pool_q) / math.sqrt(h.shape[-1])
            score = score.masked_fill(attention_mask == 0, -1e4)
            alpha = torch.softmax(score, dim=1)
            return (h * alpha.unsqueeze(-1)).sum(dim=1)

        if self.mode == "last":
            h = hs[-1]
        elif self.mode == "mean":
            h = hs.mean(dim=0)
        else:  # weighted
            w = torch.softmax(self.layer_weights, dim=0)
            h = (hs * w.view(-1, 1, 1, 1)).sum(dim=0)

        if self.pooling == "cls":
            return h[:, 0, :]
        if self.pooling == "mean":
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        # max
        m = attention_mask.unsqueeze(-1).bool()
        return h.masked_fill(~m, -1e9).max(dim=1).values


# ============================================================
# LogLLM
# ============================================================

class LogLLM(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.cfg = config
        self.device = device
        self.compute_dtype = _resolve_dtype(config.compute_dtype)

        # ---------- Encoder ----------
        enc_spec = resolve_encoder(config.encoder, config.encoder_overrides)
        self.encoder_spec = enc_spec
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(enc_spec["path"])

        enc_cfg = AutoConfig.from_pretrained(enc_spec["path"])
        if enc_spec["attention_window"] and _is_longformer_like(enc_cfg):
            win = int(enc_spec["attention_window"])
            if isinstance(enc_cfg.attention_window, int):
                enc_cfg.attention_window = [win] * enc_cfg.num_hidden_layers
            else:
                enc_cfg.attention_window = [win] * len(enc_cfg.attention_window)

        if config.encoder_quantize:
            enc_bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=True)
            self.encoder = AutoModel.from_pretrained(
                enc_spec["path"], config=enc_cfg,
                quantization_config=enc_bnb,
                device_map=device) # low_cpu_mem_usage=True,  ???
            self.encoder = prepare_model_for_kbit_training(self.encoder, use_gradient_checkpointing=False)
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.encoder = AutoModel.from_pretrained(
                enc_spec["path"], config=enc_cfg,
                torch_dtype=self.compute_dtype,
                low_cpu_mem_usage=True).to(device)

        self.encoder_dim = self.encoder.config.hidden_size
        self._longformer_like = _is_longformer_like(self.encoder.config)

        # ---------- Aggregator ----------
        self.encoder_last_layers = int(config.encoder_last_layers)
        self.encoder_layer_agg = config.encoder_layer_agg
        self.encoder_pooling = config.encoder_pooling
        self.aggregator = LayerAggregator(
            hidden=self.encoder_dim,
            k=self.encoder_last_layers,
            mode=self.encoder_layer_agg,
            pooling=self.encoder_pooling,
        ).to(device)

        # ---------- Encoder LoRA ----------
        lora_targets = (config.encoder_lora_targets
                        or _auto_lora_targets(self.encoder))
        print(f"[encoder] {enc_spec['path']} | dim={self.encoder_dim} | "
              f"K={self.encoder_last_layers} agg={self.encoder_layer_agg} "
              f"pool={self.encoder_pooling} | lora={lora_targets}")
        self.encoder = get_peft_model(self.encoder, LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.encoder_lora_r, lora_alpha=config.encoder_lora_alpha,
            lora_dropout=0.01, target_modules=lora_targets))

        # ---------- Llama QLoRA ----------
        llama_bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.compute_dtype)
        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            config.llama_path, padding_side=config.decoder_tokenizer_padding_side)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama = AutoModelForCausalLM.from_pretrained(
            config.llama_path, quantization_config=llama_bnb,
            low_cpu_mem_usage=True, device_map=device)
        self.llama = prepare_model_for_kbit_training(self.llama)
        self.llama.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

        self.llama = get_peft_model(self.llama, LoraConfig(
            r=config.llama_lora_r, lora_alpha=config.llama_lora_alpha,
            lora_dropout=config.decoder_lora_dropout,
            target_modules=config.decoder_target_modules,
            bias="none", task_type=TaskType.CAUSAL_LM))
        self.llama_dim = self.llama.config.hidden_size

        # ---------- Projector ----------
        self.projector = nn.Linear(self.encoder_dim, self.llama_dim).to(device)

        # ---------- Time fusion ----------
        self.use_time = config.use_time_features
        self.fusion_type = config.fusion_type if self.use_time else "none"
        if self.use_time and self.fusion_type in ("cross_attention", "concat"):
            self.time_encoder = TimeFeatureEncoder(
                out_dim=self.llama_dim, emb_len=config.time_emb_len,
                use_cyclic=config.use_cyclic_time,
                use_delta=config.use_time2vec,
                dtype=self.compute_dtype).to(device)
            if self.fusion_type == "cross_attention":
                self.cross_attn = CrossAttentionFusion(self.llama_dim).to(device)
            elif self.fusion_type == "concat":
                self.concat_proj = nn.Linear(2 * self.llama_dim,
                                              self.llama_dim).to(device)

        # ---------- Head ----------
        self.classifier = nn.Sequential(
            nn.Linear(self.llama_dim, 256),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1),
        ).to(device)

        self.ins1_str = "Below is a sequence of system log messages:"
        self.ins2_str = ". Is this sequence normal or anomalous?"

    # ---------- Encode ----------
    def _encode_logs(self, enc_inputs, seq_positions):
        need_hs = self.encoder_last_layers > 1
        out = self.encoder(**enc_inputs, output_hidden_states=need_hs)
        hidden = out.hidden_states if need_hs else (out.last_hidden_state,)
        pooled = self.aggregator(hidden, enc_inputs["attention_mask"])
        proj = self.projector(pooled.float()).to(self.compute_dtype)
        return list(torch.tensor_split(proj, seq_positions)) if seq_positions else [proj]

    def _apply_fusion(self, window_embs, times_list):
        if not self.use_time or self.fusion_type in ("projector", "none"):
            return window_embs

        B = len(window_embs)
        Ls = [e.shape[0] for e in window_embs]
        L_max = max(Ls) if Ls else 1
        D = window_embs[0].shape[1]

        padded = torch.zeros(B, L_max, D, dtype=self.compute_dtype,
                             device=self.device)
        mask = torch.zeros(B, L_max, dtype=torch.long, device=self.device)
        for b, e in enumerate(window_embs):
            n = e.shape[0]
            padded[b, :n] = e
            mask[b, :n] = 1

        time_emb = self.time_encoder(times_list)  # fp32
        padded_fp32 = padded.float()  # ← каст для fusion

        if self.fusion_type == "cross_attention":
            fused = self.cross_attn(time_emb, padded_fp32, log_mask=mask)
        elif self.fusion_type == "concat":
            fused = self.concat_proj(torch.cat([padded_fp32, time_emb], dim=-1))
        else:
            fused = padded_fp32

        fused = fused.to(self.compute_dtype)  # ← обратно в bf16
        return [fused[b, :Ls[b]] for b in range(B)]

    # ---------- Forward ----------
    def forward(self, batch):
        enc_inputs = {k: v.to(self.device)
                      for k, v in batch["bert_inputs"].items()}

        if self._longformer_like:
            g = torch.zeros_like(enc_inputs["attention_mask"])
            g[:, 0] = 1
            enc_inputs["global_attention_mask"] = g

        seq_positions = batch["seq_positions"].tolist()
        times_list = batch["times"]
        B = len(times_list)

        window_embs = self._encode_logs(enc_inputs, seq_positions)
        window_embs = self._apply_fusion(window_embs, times_list)

        embed = self.llama.get_input_embeddings()
        ins1 = embed(self.llama_tokenizer(
            self.ins1_str, return_tensors="pt",
            add_special_tokens=False)["input_ids"][0].to(self.device)
                     ).to(self.compute_dtype)
        ins2 = embed(self.llama_tokenizer(
            self.ins2_str, return_tensors="pt",
            add_special_tokens=False)["input_ids"][0].to(self.device)
                     ).to(self.compute_dtype)

        seqs = [torch.cat([ins1, e, ins2], dim=0) for e in window_embs]
        max_len = max(s.shape[0] for s in seqs)
        D = seqs[0].shape[1]
        padded = torch.zeros(B, max_len, D, dtype=self.compute_dtype,
                             device=self.device)
        mask = torch.zeros(B, max_len, dtype=torch.long, device=self.device)
        for b, s in enumerate(seqs):
            n = s.shape[0]
            padded[b, :n] = s
            mask[b, :n] = 1

        out = self.llama(inputs_embeds=padded, attention_mask=mask,
                         output_hidden_states=False)
        # hidden = out.hidden_states[-1]
        hidden = out.last_hidden_state

        if self.cfg.decoder_head_pooling == "last":

            if self.cfg.decoder_tokenizer_padding_side == "left":
                last_idx = torch.full((B,), max_len-1, device=mask.device)
            else:
                last_idx = mask.sum(dim=1) - 1

            last_hidden = hidden[torch.arange(B, device=self.device), last_idx]
        elif self.cfg.decoder_head_pooling == "mean":
            # bs, sl, hs x  bs, sl, 1
            last_hidden = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return self.classifier(last_hidden.float()).squeeze(-1)

    # ---------- Train modes ----------
    def set_train_mode(self, mode: str):
        """'llama' | 'head' | 'encoder' | 'all'"""
        for p in self.parameters():
            p.requires_grad = False

        if mode == "llama":
            for n, p in self.llama.named_parameters():
                if "lora" in n:
                    p.requires_grad = True
            return

        # head: projector + classifier + aggregator (+ time fusion)
        for m in (self.projector, self.classifier, self.aggregator):
            for p in m.parameters():
                p.requires_grad = True
        if self.use_time and self.fusion_type in ("cross_attention", "concat"):
            for p in self.time_encoder.parameters():
                p.requires_grad = True
            if self.fusion_type == "cross_attention":
                for p in self.cross_attn.parameters():
                    p.requires_grad = True
            elif self.fusion_type == "concat":
                for p in self.concat_proj.parameters():
                    p.requires_grad = True

        if mode in ("encoder", "all"):
            for n, p in self.encoder.named_parameters():
                if "lora" in n:
                    p.requires_grad = True
        if mode == "all":
            for n, p in self.llama.named_parameters():
                if "lora" in n:
                    p.requires_grad = True

    # ---------- Save / Load ----------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.encoder.save_pretrained(os.path.join(path, "encoder_lora"))
        self.llama.save_pretrained(os.path.join(path, "llama_lora"))
        for name in ("projector", "classifier", "aggregator"):
            torch.save(getattr(self, name).state_dict(),
                       os.path.join(path, f"{name}.pt"))
        if self.use_time and self.fusion_type in ("cross_attention", "concat"):
            torch.save(self.time_encoder.state_dict(),
                       os.path.join(path, "time_encoder.pt"))
            name = ("cross_attn" if self.fusion_type == "cross_attention"
                    else "concat_proj")
            torch.save(getattr(self, name).state_dict(),
                       os.path.join(path, f"{name}.pt"))

    def load(self, path: str):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        for sub, model in (("encoder_lora", self.encoder),
                            ("llama_lora", self.llama)):
            d = os.path.join(path, sub)
            if not os.path.isdir(d):
                continue
            state = None
            for fn in ("adapter_model.safetensors", "adapter_model.bin"):
                fp = os.path.join(d, fn)
                if os.path.exists(fp):
                    state = (load_file(fp) if fn.endswith(".safetensors")
                             else torch.load(fp, map_location="cpu"))
                    break
            if state is not None:
                set_peft_model_state_dict(model, state)

        for name in ("projector", "classifier", "aggregator"):
            fp = os.path.join(path, f"{name}.pt")
            if os.path.exists(fp):
                getattr(self, name).load_state_dict(
                    torch.load(fp, map_location=self.device))

        if self.use_time and self.fusion_type in ("cross_attention", "concat"):
            fp = os.path.join(path, "time_encoder.pt")
            if os.path.exists(fp):
                self.time_encoder.load_state_dict(
                    torch.load(fp, map_location=self.device))
            name = ("cross_attn" if self.fusion_type == "cross_attention"
                    else "concat_proj")
            fp = os.path.join(path, f"{name}.pt")
            if os.path.exists(fp):
                getattr(self, name).load_state_dict(
                    torch.load(fp, map_location=self.device))
"""
Ядро: временные признаки, Jasper, кэш эмбеддингов, проектор, агрегатор,
Longformer + LoRA + опциональная квантизация, метрики, save/load.
"""
import os
import math
import typing
import hashlib
import contextlib
import datetime as _dt
from collections import OrderedDict
import tqdm.auto as tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    LongformerModel, LongformerConfig, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_recall_fscore_support


# ============================================================
# 1. Временные признаки
# ============================================================

class Time2Vec(nn.Module):
    """t2v(t)[0] = w0·t + b0; t2v(t)[i] = sin(w_i·t + b_i)."""
    def __init__(self, k: int = 16):
        super().__init__()
        self.k  = k
        self.w0 = nn.Parameter(torch.randn(1) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w  = nn.Parameter(torch.randn(k) * 0.01)
        self.b  = nn.Parameter(torch.zeros(k))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.squeeze(-1)
        lin = (self.w0 * t + self.b0).unsqueeze(-1)          # [N,1]
        per = torch.sin(t.unsqueeze(-1) * self.w + self.b)   # [N,k]
        return torch.cat([lin, per], dim=-1)                 # [N,k+1]


def cyclic_time_features(times: typing.List[_dt.datetime]) -> torch.Tensor:
    feats = []
    for t in times:
        if t is None:
            feats.append([0.0] * 10)
            continue
        h   = t.hour / 24.0
        m   = t.minute / 60.0
        dow = t.weekday() / 7.0
        mon = (t.month - 1) / 12.0
        dom = (t.day - 1) / 31.0
        feats.append([
            math.sin(2 * math.pi * h),   math.cos(2 * math.pi * h),
            math.sin(2 * math.pi * m),   math.cos(2 * math.pi * m),
            math.sin(2 * math.pi * dow), math.cos(2 * math.pi * dow),
            math.sin(2 * math.pi * mon), math.cos(2 * math.pi * mon),
            math.sin(2 * math.pi * dom), math.cos(2 * math.pi * dom),
        ])
    return torch.tensor(feats, dtype=torch.float32)


def time_delta_features(times: typing.List[_dt.datetime]) -> torch.Tensor:
    deltas = []
    for i, t in enumerate(times):
        if i == 0 or t is None or times[i - 1] is None:
            deltas.append(0.0)
        else:
            deltas.append(float((t - times[i - 1]).total_seconds()))
    return torch.tensor(deltas, dtype=torch.float32).unsqueeze(-1)  # [N,1]


# ============================================================
# 2. Jasper encoder (заморожен)
# ============================================================

def load_jasper_encoder(model_name: str, compression_ratio: float,
                        device: str = "cuda"):
    model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def encode(texts: typing.List[str]) -> torch.Tensor:
        emb = model.encode(
            texts,
            compression_ratio=compression_ratio,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
        return emb.to(torch.bfloat16)

    return encode


# ============================================================
# 3. LRU-кэш эмбеддингов строк
# ============================================================

class LineEmbeddingCache:
    """LRU-кэш на уровне строк. Хранит CPU-тензоры bf16."""
    def __init__(self, encode_fn, max_lines: int = 50_000,
                 encode_batch: int = 64):
        self.encode_fn = encode_fn
        self.max_lines = max_lines
        self.encode_batch = encode_batch
        self.cache: "OrderedDict[bytes, torch.Tensor]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(line: str) -> bytes:
        return hashlib.blake2b(line.encode("utf-8", "replace"),
                               digest_size=16).digest()

    @torch.no_grad()
    def get_batch(self, lines: typing.List[str]) -> torch.Tensor:
        n = len(lines)
        out: typing.List[typing.Optional[torch.Tensor]] = [None] * n
        miss_i: typing.List[int] = []
        miss_lines: typing.List[str] = []

        for i, line in enumerate(lines):
            k = self._key(line)
            emb = self.cache.get(k)
            if emb is not None:
                self.cache.move_to_end(k)
                out[i] = emb
                self.hits += 1
            else:
                miss_i.append(i)
                miss_lines.append(line)

        if miss_lines:
            self.misses += len(miss_lines)
            new_embs = []
            for j in range(0, len(miss_lines), self.encode_batch):
                chunk = miss_lines[j:j + self.encode_batch]
                emb = self.encode_fn(chunk)                       # GPU bf16
                new_embs.append(emb.detach().to("cpu"))
            new_embs = torch.cat(new_embs, dim=0)
            for j, i in enumerate(miss_i):
                k = self._key(miss_lines[j])
                out[i] = new_embs[j]
                self.cache[k] = new_embs[j]
                self.cache.move_to_end(k)
            overflow = len(self.cache) - self.max_lines
            for _ in range(max(0, overflow)):
                self.cache.popitem(last=False)

        return torch.stack(out, dim=0)                            # [N,D] CPU

    def stats(self) -> str:
        t = self.hits + self.misses
        hr = self.hits / t * 100 if t else 0.0
        return (f"cache size={len(self.cache)}/{self.max_lines} "
                f"hits={self.hits} misses={self.misses} hit_rate={hr:.1f}%")


def make_collate_fn(cache: LineEmbeddingCache):
    """DataLoader.collate_fn: кодирует окна, паддит, собирает маску."""
    def collate(batch):
        windows, times_list, labels = [], [], []
        for window, window_times, _raws, label in batch:
            if not window:
                continue
            emb = cache.get_batch(window)                         # [n_i,D] CPU
            windows.append(emb)
            times_list.append(window_times)
            labels.append(float(label))

        if not windows:
            return None
        max_len = max(e.shape[0] for e in windows)
        D = windows[0].shape[1]
        padded = torch.zeros(len(windows), max_len, D, dtype=torch.bfloat16)
        mask   = torch.zeros(len(windows), max_len, dtype=torch.long)
        for i, e in enumerate(windows):
            n = e.shape[0]
            padded[i, :n] = e
            mask[i, :n]   = 1

        return {
            "log_embs": padded,
            "attention_mask": mask,
            "times": times_list,
            "labels": torch.tensor(labels, dtype=torch.float32),
        }
    return collate


# ============================================================
# 4. Проектор Jasper → Longformer
# ============================================================

class LogProjector(nn.Module):
    """
    LayerNorm → Linear → GELU → Linear, последний Linear — zero-init,
    residual через отдельный Linear при разных размерностях.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_mult: int = 2):
        super().__init__()
        hidden = out_dim * hidden_mult
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.residual = nn.Identity() if in_dim == out_dim \
                        else nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        return self.residual(x) + self.net(x)


# ============================================================
# 5. Агрегатор по слоям Longformer
# ============================================================

class LayerAggregator(nn.Module):
    """Агрегация последних K слоёв: mean | weighted | attn_pool | last."""
    def __init__(self, hidden: int, k: int = 4, mode: str = "weighted"):
        super().__init__()
        self.k = k
        self.mode = mode
        if mode == "weighted":
            self.layer_weights = nn.Parameter(torch.zeros(k))
        elif mode == "attn_pool":
            self.pool_q = nn.Parameter(torch.randn(hidden) * 0.02)

    def forward(self, hidden_states, attention_mask):
        hs = torch.stack(hidden_states[-self.k:], dim=0)          # [K,B,L,H]
        if self.mode == "mean":
            h = hs.mean(dim=0)
        elif self.mode == "weighted":
            w = torch.softmax(self.layer_weights, dim=0)
            h = (hs * w.view(-1, 1, 1, 1)).sum(dim=0)
        elif self.mode == "last":
            h = hs[-1]
        else:
            h = hs.mean(dim=0)                                     # fallback

        if self.mode == "attn_pool":
            score = (h @ self.pool_q) / math.sqrt(h.shape[-1])
            score = score.masked_fill(attention_mask == 0, -1e4)
            alpha = torch.softmax(score, dim=1)
            return (h * alpha.unsqueeze(-1)).sum(dim=1)

        m = attention_mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


# ============================================================
# 6. Dtype / квантизация
# ============================================================

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
    "float32":  torch.float32,
}


def resolve_dtype(cfg) -> torch.dtype:
    if cfg.use_quantization:
        return _DTYPE_MAP.get(cfg.bnb_compute_dtype, torch.bfloat16)
    return _DTYPE_MAP.get(cfg.mixed_dtype, torch.bfloat16)


def build_bnb_config(cfg) -> typing.Optional[BitsAndBytesConfig]:
    if not cfg.use_quantization:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=cfg.bnb_4bit,
        load_in_8bit=not cfg.bnb_4bit,
        bnb_4bit_quant_type=cfg.bnb_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_double_quant,
        bnb_4bit_compute_dtype=_DTYPE_MAP[cfg.bnb_compute_dtype],
    )


def build_longformer_with_lora(cfg, device: torch.device) -> PeftModel:
    lf_cfg = LongformerConfig.from_pretrained(cfg.longformer_model_name)
    lf_cfg.num_hidden_layers = cfg.longformer_layers
    lf_cfg.attention_window = [cfg.longformer_attention_window] * cfg.longformer_layers

    kwargs = dict(config=lf_cfg)
    if cfg.use_quantization:
        kwargs["quantization_config"] = build_bnb_config(cfg)
        kwargs["device_map"] = {"": device.index if device.index is not None else 0}
    else:
        kwargs["torch_dtype"] = resolve_dtype(cfg)

    base = LongformerModel.from_pretrained(cfg.longformer_model_name, **kwargs)
    if not cfg.use_quantization:
        base = base.to(device)

    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    return get_peft_model(base, lora_cfg)


# ============================================================
# 7. Основная модель
# ============================================================

class LogAnomalyModel(nn.Module):
    def __init__(self, cfg, jasper_dim: int, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.longformer = build_longformer_with_lora(cfg, device)
        lf_hidden = self.longformer.config.hidden_size
        self.lf_hidden = lf_hidden

        self.projector  = LogProjector(jasper_dim, lf_hidden)
        self.aggregator = LayerAggregator(
            lf_hidden, k=cfg.aggregate_layers, mode=cfg.aggregate_mode,
        )
        self.time2vec = Time2Vec(cfg.time_emb_len) if cfg.use_time2vec else None

        self.use_cls = cfg.use_learned_cls
        if self.use_cls:
            self.cls_emb = nn.Parameter(torch.randn(1, 1, lf_hidden) * 0.02)

        time_dim = (cfg.time_emb_len + 1) if cfg.use_time2vec else 0
        cyclic_dim = 10 if cfg.use_cyclic_time else 0
        has_time_dim = 1 if getattr(cfg, "use_time_present", False) else 0
        head_dim = lf_hidden + time_dim + cyclic_dim + 1 + has_time_dim

        self.classifier = nn.Sequential(
            nn.Linear(head_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def trainable_parameters(self):
        """Параметры, которые надо сохранять в чекпоинт (всё, кроме base LF)."""
        head_names = {"projector", "aggregator", "time2vec",
                      "classifier", "cls_emb"}
        for name, p in self.named_parameters():
            root = name.split(".")[0]
            if root in head_names and p.requires_grad:
                yield name, p

    def head_state_dict(self) -> dict:
        sd = {}
        for name, p in self.trainable_parameters():
            sd[name] = p.detach().cpu()
        return sd

    def load_head_state_dict(self, sd: dict):
        own = dict(self.named_parameters())
        for name, tensor in sd.items():
            if name in own:
                own[name].data.copy_(tensor.to(own[name].device,
                                                dtype=own[name].dtype))

    def forward(self, log_embs, times, attention_mask):
        arget_dtype = next(self.projector.parameters()).dtype
        log_embs = log_embs.to(target_dtype)

        x = self.projector(log_embs)                              # [B,L,H]
        B, L, _ = x.shape

        if self.use_cls:
            cls = self.cls_emb.expand(B, -1, -1).to(x.dtype)
            x = torch.cat([cls, x], dim=1)                        # [B,L+1,H]
            ones = torch.ones(B, 1, dtype=attention_mask.dtype,
                              device=x.device)
            attention_mask = torch.cat([ones, attention_mask], dim=1)

        original_len = attention_mask.shape[1]  # до вызова longformer

        lf_out = self.longformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden = lf_out.hidden_states
        # Longformer паддит вход до кратного attention_window — режем обратно
        hidden = tuple(h[:, :original_len, :] for h in hidden)

        K = self.aggregator.k

        if self.use_cls:
            hs = torch.stack([h[:, 0] for h in hidden[-K:]], dim=0)  # [K,B,H]
            if self.aggregator.mode == "weighted":
                w = torch.softmax(self.aggregator.layer_weights, dim=0)
                pooled = (hs * w.view(-1, 1, 1)).sum(dim=0)
            else:
                pooled = hs.mean(dim=0)
        else:
            pooled = self.aggregator(hidden, attention_mask)

        # Временные признаки (усреднённые по окну)
        time_feats = []
        for b in range(B):
            t_list = times[b]
            delta = time_delta_features(t_list).to(pooled.device, pooled.dtype)

            parts = []
            if self.time2vec is not None:
                t2v = self.time2vec(delta.squeeze(-1))
                parts.append(t2v.mean(dim=0))
            if self.cfg.use_cyclic_time:
                cyc = cyclic_time_features(t_list).to(pooled.device, pooled.dtype)
                parts.append(cyc.mean(dim=0))
            parts.append(delta.mean(dim=0))

            if getattr(self.cfg, "use_time_present", False):
                has_time = torch.tensor(
                    [1.0 if t is not None else 0.0 for t in t_list],
                    device=pooled.device, dtype=pooled.dtype,
                ).mean().unsqueeze(0)
                parts.append(has_time)

            time_feats.append(torch.cat(parts))

        time_feats = torch.stack(time_feats)

        combined = torch.cat([pooled, time_feats], dim=-1)
        logits = self.classifier(combined).squeeze(-1)
        return logits


# ============================================================
# 8. Autocast helper
# ============================================================

def autocast_context(cfg):
    if not cfg.use_autocast or not torch.cuda.is_available():
        return contextlib.nullcontext()
    if cfg.use_quantization:
        # bnb сам управляет compute dtype — внешний autocast не нужен
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=resolve_dtype(cfg))


# ============================================================
# 9. Метрики
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, criterion, autocast_ctx=None,
             max_windows: typing.Optional[int] = None, max_batches=None, desc="eval") -> dict:
    model.eval()
    losses, probs, ys = [], [], []
    n_seen = 0
    if autocast_ctx is None:
        autocast_ctx = contextlib.nullcontext()

    pbar = tqdm.tqdm(loader, desc=desc, leave=False)
    for i, batch in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break

        if batch is None:
            continue
        log_embs = batch["log_embs"].to(device)
        mask     = batch["attention_mask"].to(device)
        y        = batch["labels"].to(device)

        with autocast_ctx:
            logits = model(log_embs, batch["times"], mask)
        losses.append(criterion(logits.float(), y).item())
        probs.append(torch.sigmoid(logits.float()).cpu())
        ys.append(y.cpu())
        n_seen += y.numel()

        pbar.set_postfix({"seen": n_seen})
        if max_windows is not None and n_seen >= max_windows:
            break

    if not losses:
        return {"loss": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_windows": 0, "n_pos": 0}

    p_prob = torch.cat(probs).numpy()
    y_true = torch.cat(ys).numpy().astype(int)
    y_pred = (p_prob >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "loss": float(np.mean(losses)),
        "precision": float(pr),
        "recall": float(rc),
        "f1": float(f1),
        "n_windows": int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }


# ============================================================
# 10. Save / load trainable state
# ============================================================

def save_trainable(model, optimizer, scheduler, epoch, global_step, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.longformer.save_pretrained(os.path.join(save_dir, "adapter"))
    torch.save({
        "head_state": model.head_state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "epoch":      epoch,
        "global_step": global_step,
    }, os.path.join(save_dir, "train_state.pt"))


def load_adapter_into(model, adapter_dir: str):
    from peft import PeftModel
    if isinstance(model.longformer, PeftModel):
        # уже PeftModel — просто грузим поверх
        try:
            model.longformer.load_adapter(adapter_dir, adapter_name="default")
            model.longformer.set_adapter("default")
        except Exception:
            # fallback: пересоздаём
            model.longformer = PeftModel.from_pretrained(
                model.longformer.base_model, adapter_dir, is_trainable=True
            )
    else:
        model.longformer = PeftModel.from_pretrained(
            model.longformer, adapter_dir, is_trainable=True
        )
    for n, p in model.longformer.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)

def load_trainable(model, optimizer, scheduler, load_dir, device):
    ckpt = torch.load(os.path.join(load_dir, "train_state.pt"),
                      map_location="cpu")
    model.load_head_state_dict(ckpt["head_state"])
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    except Exception as e:
        print(f"[resume] не удалось восстановить optimizer/scheduler: {e}")
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0)
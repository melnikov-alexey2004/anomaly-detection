import os, math, typing, dataclasses, datetime, random
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer, AutoModel, LongformerModel, LongformerConfig,
    get_linear_schedule_with_warmup, Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from sentence_transformers import SentenceTransformer
from huggingface_hub import notebook_login
from run import Config



class Time2Vec(nn.Module):
    """
    t2v(τ)[0] = w0·τ + b0
    t2v(τ)[i] = sin(w_i·τ + b_i),  i = 1..k
    """
    def __init__(self, k: int = 16):
        super().__init__()
        self.k = k
        self.w0 = nn.Parameter(torch.randn(1) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w  = nn.Parameter(torch.randn(k) * 0.01)
        self.b  = nn.Parameter(torch.zeros(k))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [N] или [N, 1]
        if t.dim() == 2:
            t = t.squeeze(-1)
        lin = (self.w0 * t + self.b0).unsqueeze(-1)          # [N, 1]
        per = torch.sin(t.unsqueeze(-1) * self.w + self.b)   # [N, k]
        return torch.cat([lin, per], dim=-1)                 # [N, k+1]

def cyclic_time_features(times: typing.List[datetime.datetime]) -> torch.Tensor:
    """
    Принимает список datetime, возвращает тензор [N, 10]:
    sin/cos для часа, минуты, дня недели, месяца, дня месяца.
    """
    feats = []
    for t in times:
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

def time_delta_features(times: typing.List[datetime.datetime]) -> torch.Tensor:
    """
    Возвращает [N-1] — разности в секундах между соседними записями.
    Первое значение дублируется, чтобы длина совпадала с окном.
    """
    deltas = []
    for i in range(len(times)):
        if i == 0:
            deltas.append(0.0)
        else:
            d = (times[i] - times[i - 1]).total_seconds()
            deltas.append(d)
    return torch.tensor(deltas, dtype=torch.float32).unsqueeze(-1)  # [N, 1]


class LogAnomalyModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        jasper_dim = cfg.log_encoder_emb_dim

        # --- Longformer ---
        lf_config = LongformerConfig.from_pretrained(cfg.longformer_model_name)
        lf_config.num_hidden_layers = cfg.longformer_layers
        lf_config.attention_window = [cfg.longformer_attention_window] * cfg.longformer_layers
        self.longformer = LongformerModel.from_pretrained(
            cfg.longformer_model_name, config=lf_config
        )

        # --- LoRA ---
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            inference_mode=False,
        )

        self.longformer.train()
        for named, params in self.longformer.named_parameters():
            params.requires_grad = False
        self.longformer = get_peft_model(self.longformer, lora_cfg)

        # --- временные слои ---
        self.time2vec = Time2Vec(cfg.time_emb_len) if cfg.use_time2vec else None
        time_dim = (cfg.time_emb_len + 1) if cfg.use_time2vec else 0
        cyclic_dim = 10 if cfg.use_cyclic_time else 0
        delta_dim = 1

        # --- проекция эмбеддингов лога в размерность Longformer ---
        self.log_proj = nn.Linear(jasper_dim, lf_config.hidden_size)

        # --- классификатор ---
        total_dim = lf_config.hidden_size + time_dim + cyclic_dim + delta_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        log_embs: torch.Tensor,          # [B, N, jasper_dim]
        times: typing.List[typing.List[datetime.datetime]],
        attention_mask: torch.Tensor,    # [B, N]
    ) -> torch.Tensor:
        B, N, _ = log_embs.shape

        # 1. Проекция эмбеддингов лога
        log_embs = self.log_proj(log_embs)                     # [B, N, H]

        # 2. Longformer
        lf_out = self.longformer(
            inputs_embeds=log_embs,
            attention_mask=attention_mask,
        ).last_hidden_state                                    # [B, N, H]

        # 3. Усреднение по токенам (можно заменить на CLS-токен)
        pooled = lf_out.mean(dim=1)                            # [B, H]

        # 4. Временные признаки
        time_feats = []
        for b in range(B):
            t_list = times[b]
            delta = time_delta_features(t_list)                # [N, 1]
            cyclic = cyclic_time_features(t_list) if self.cfg.use_cyclic_time else None
            t2v = self.time2vec(delta.squeeze(-1)) if self.time2vec else None

            parts = []
            if t2v is not None:
                parts.append(t2v.mean(dim=0))                  # [k+1]
            if cyclic is not None:
                parts.append(cyclic.mean(dim=0))               # [10]
            parts.append(delta.mean(dim=0))                    # [1]
            time_feats.append(torch.cat(parts))                # [k+1+10+1]

        time_feats = torch.stack(time_feats)                   # [B, time_dim]
        combined = torch.cat([pooled, time_feats], dim=-1)     # [B, H + time_dim]
        logits = self.classifier(combined)                     # [B, 1]
        return logits.squeeze(-1)

class LogWindowDataset(Dataset):
    #todo:
    raise NotImplementedError







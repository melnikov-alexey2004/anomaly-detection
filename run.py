import dataclasses
import typing
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
from dataset import BGL, Tbird, Liberty, Spirit, EvaluationDataset, SuperComputerDataset,\
    BalancedSampler, make_collate_fn
from model import LogAnomalyModel

notebook_login()   # вводит токен и сохраняет его в ~/.huggingface

@dataclasses.dataclass
class Config:
    # ---- данные ----
    win_size: int = 200            # длина окна (число записей лога)
    step_size: int = 200           # шаг скольжения окна
    batch_size: int = 16
    train_ratio: float = 0.3       # доля обучающей выборки
    max_lines = math.inf

    # ---- временные признаки ----
    time_emb_len: int = 16         # размерность Time2Vec (k)
    use_time2vec: bool = True ###
    use_cyclic_time: bool = True

    # ---- эмбеддинги лога ----
    jasper_model_name: str = "infgrad/Jasper-Token-Compression-600M"
    compression_ratio: float = 0.5  # оптимальное значение из диапазона 0.3–0.8

    # ---- Longformer ----
    longformer_model_name: str = "allenai/longformer-base-4096"
    longformer_layers: int = 12              # число слоёв (границы: 4–24)
    longformer_attention_window: int = 512   # размер окна внимания (границы: 128–4096)
    max_seq_len: int = 4096                  # максимальная длина последовательности

    # ---- LoRA ----
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: typing.List[str] = dataclasses.field(
        default_factory=lambda: ["query", "key", "value", "dense"]
    )

    # ---- обучение ----
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1
    scheduler_type: str = "linear"   # "linear" или "cosine", но НЕ "exponential"
    sampler_seed: int = 42

    # ---- сохранение ----
    hub_model_id: str = "your-username/log-anomaly-longformer"
    push_to_hub: bool = True

    target_ratio: float = 0.3  # для BalancedSampler
    jasper_batch: int = 32  ### размер чанка при кодировании окна Jasper'ом

    # утсанавливается в процессе run()
    log_encoder_emb_dim: int = 1 ######

def load_jasper_encoder(model_name: str, compression_ratio: float, cfg: Config, device: str = "cuda"):
    model = SentenceTransformer(
        model_name,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )
    # === явная заморозка ===
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    cfg.log_encoder_emb_dim = model.get_sentence_embedding_dimension()

    @torch.no_grad()
    def encode(texts: typing.List[str]) -> torch.Tensor:
        emb = model.encode(
            texts,
            compression_ratio=compression_ratio,
            normalize_embeddings=True,
            convert_to_tensor=True,
            # batch_size=cfg.jasper_batch,  # подобранное значение
        )
        return emb.float() # todo:  # .float() — важно, если модель в bf16, а Longformer в fp32

    return encode


def build_source(name, cfg):
    cls = {"bgl": BGL, "tbird": Tbird, "spirit": Spirit, "liberty": Liberty}[name]
    return cls(window_size=cfg.win_size, step_size=cfg.step_size, use_in_colab=True, max_lines=cfg.max_lines)

# посмотреть шедулеры из трансофрмерс
# get_linear_schedule_with_warmup
# CosineAnnealingLR
from dataset import SuperComputerDataset
def run(dataset: typing.Literal["bgl", "tbird", "spirit", "liberty"], cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) источник + датасет (ваш)
    d  = build_source(dataset, cfg)
    ds = SuperComputerDataset(d, n=cfg.step_size, train_ratio=cfg.train_ratio)

    # 2) энкодер Jasper (заморожен)
    encode_fn = load_jasper_encoder(
        cfg.jasper_model_name, cfg.compression_ratio, device=str(device)
    )
    jasper_dim = encode_fn(["probe"]).shape[-1]

    # 3) сэмплеры / загрузчики
    train_sampler = BalancedSampler(ds, target_ratio=cfg.target_ratio, seed=42)
    val_ds        = EvaluationDataset(ds)

    collate = make_collate_fn(encode_fn, jasper_batch=cfg.jasper_batch)

    train_loader = DataLoader(
        ds, batch_size=cfg.batch_size,
        sampler=train_sampler,          # <- ваш сэмплер
        collate_fn=collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size,
        shuffle=False, collate_fn=collate, num_workers=0,
    )

    # 4) модель
    model = LogAnomalyModel(cfg, jasper_dim).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    total_steps = len(train_sampler) // cfg.batch_size * cfg.num_epochs
    if cfg.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup_steps,
            num_training_steps=total_steps,
        )

    criterion = nn.BCEWithLogitsLoss()

    # 5) обучение
    model.train()
    for epoch in range(cfg.num_epochs):
        for batch in train_loader:
            log_embs = batch["log_embs"].to(device)
            mask     = batch["attention_mask"].to(device)
            y        = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(log_embs, batch["times"], mask)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # валидация (IterableDataset — пройти заново)
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch["log_embs"].to(device),
                    batch["times"],
                    batch["attention_mask"].to(device),
                )
                tot += criterion(logits, batch["labels"].to(device)).item()
                n += 1
        print(f"[{epoch+1}] val_loss={tot/max(n,1):.4f}")
        model.train()

    # 6) сохранение
    if cfg.push_to_hub:
        model.longformer.save_pretrained(cfg.hub_model_id)
        AutoTokenizer.from_pretrained(cfg.longformer_model_name)\
                     .save_pretrained(cfg.hub_model_id)

if __name__ == "__main__":
    cfg = Config(
        win_size=200,
        step_size=200,
        batch_size=16,
        train_ratio=0.3,
        time_emb_len=16,
        jasper_model_name="infgrad/Jasper-Token-Compression-600M",
        compression_ratio=0.5,
        longformer_layers=12,
        longformer_attention_window=512,
        lora_r=8,
        lora_alpha=16,
        num_epochs=3,
        scheduler_type="linear",
        hub_model_id="your-username/bgl-anomaly-longformer",
    )

    run("bgl", cfg)
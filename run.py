"""
Config + run(). Клонируем репо → запускаем run() → создаётся HF-репо,
туда кладутся LoRA-адаптеры и метрики. Следующий run() может дообучиться
с прошлого чекпоинта, добавив K% данных.
"""
import os
import sys
import json
import math
import time
import random
import pickle
import typing
import dataclasses
import datetime as _dt
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from huggingface_hub import (
    notebook_login, whoami, HfApi, hf_hub_download, snapshot_download,
)

from dataset import (
    BGL, Tbird, Liberty, Spirit,
    SuperComputerDataset, EvaluationDataset, BalancedSampler,
)
from model import (
    load_jasper_encoder, LineEmbeddingCache, make_collate_fn,
    LogAnomalyModel, autocast_context, evaluate,
    save_trainable, load_trainable, load_adapter_into,
    resolve_dtype,
)


# ============================================================
# Config
# ============================================================

@dataclasses.dataclass
class Config:
    # ---------- данные ----------
    dataset: str = "bgl"                       # bgl | tbird | spirit | liberty
    win_size: int = 200
    step_size: int = 200
    anchor_step: int = 10
    batch_size: int = 16
    num_workers: int = 0
    use_in_colab: bool = True
    max_lines: typing.Optional[int] = None     # None = без лимита

    # ---------- доли ----------
    eval_start_ratio: float = 0.9              # фиксированная граница eval
    train_ratio: float = 0.3                   # сколько лога реально обучаем
    target_ratio: float = 0.3                  # доля аномалий в BalancedSampler

    # ---------- eval ----------
    eval_max_windows: int = 2000               # мелкий eval во время обучения
    metrics_log_every: int = 100               # шаг периодического eval
    final_full_eval: bool = True               # полный eval после последней эпохи

    # ---------- временные признаки ----------
    time_emb_len: int = 16
    use_time2vec: bool = True
    use_cyclic_time: bool = True

    # ---------- Jasper ----------
    jasper_model_name: str = "infgrad/Jasper-Token-Compression-600M"
    compression_ratio: float = 0.5
    jasper_batch: int = 64
    cache_max_lines: int = 50_000

    # ---------- Longformer ----------
    longformer_model_name: str = "allenai/longformer-base-4096"
    longformer_layers: int = 12                # 4..24
    longformer_attention_window: int = 512     # 128..4096

    # ---------- агрегация ----------
    aggregate_layers: int = 4
    aggregate_mode: str = "weighted"           # weighted | mean | attn_pool | last
    use_learned_cls: bool = True

    # ---------- квантизация ----------
    use_quantization: bool = False
    bnb_4bit: bool = True
    bnb_quant_type: str = "nf4"
    bnb_double_quant: bool = True
    bnb_compute_dtype: str = "bfloat16"
    mixed_dtype: str = "bfloat16"              # bfloat16 | float16 | float32
    use_autocast: bool = False                 # True = fp32-веса + autocast

    # ---------- LoRA ----------
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: typing.List[str] = dataclasses.field(
        default_factory=lambda: ["query", "key", "value", "dense"]
    )

    # ---------- обучение ----------
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    scheduler_type: str = "linear"             # linear | cosine
    grad_clip: float = 1.0
    seed: int = 42

    # ---------- Hub ----------
    push_to_hub: bool = True
    hub_repo_id: typing.Optional[str] = None   # None → <user>/<dataset>-log-anomaly
    run_tag: typing.Optional[str] = None       # None → авто
    resume_from_repo: typing.Optional[str] = None   # HF repo id с прошлым run
    resume_from_run_tag: typing.Optional[str] = None


# ============================================================
# Хелперы Config ↔ JSON
# ============================================================

def cfg_to_json_dict(cfg: Config) -> dict:
    d = dataclasses.asdict(cfg)
    for k, v in list(d.items()):
        if isinstance(v, torch.dtype):
            d[k] = str(v)
    return d


def cfg_to_json(cfg: Config) -> str:
    return json.dumps(cfg_to_json_dict(cfg), indent=2, ensure_ascii=False)


# ============================================================
# Сэмплер с фильтром (треним только < train_limit)
# ============================================================

class FilteredSampler(Sampler):
    def __init__(self, base: Sampler, max_start: int):
        self.base = base
        self.max_start = max_start

    def __iter__(self):
        for idx in self.base:
            if int(idx) < self.max_start:
                yield int(idx)

    def __len__(self):
        return len(self.base)


class LimitedIterable(torch.utils.data.IterableDataset):
    """Обрезает любой IterableDataset до max_items."""
    def __init__(self, base, max_items: typing.Optional[int]):
        self.base = base
        self.max_items = max_items

    def __iter__(self):
        if self.max_items is None:
            yield from self.base
            return
        for i, x in enumerate(self.base):
            if i >= self.max_items:
                break
            yield x

    def __len__(self):
        if self.max_items is None:
            return len(self.base)
        return min(len(self.base), self.max_items)


# ============================================================
# HF-репо
# ============================================================

def ensure_login():
    try:
        whoami()
    except Exception:
        notebook_login()


def resolve_repo_id(cfg: Config) -> str:
    if cfg.hub_repo_id:
        return cfg.hub_repo_id
    user = whoami()["name"]
    return f"{user}/{cfg.dataset}-log-anomaly"


def ensure_repo(repo_id: str):
    HfApi().create_repo(repo_id, repo_type="model", exist_ok=True)


def upload_file(local_path: str, repo_id: str, path_in_repo: str):
    HfApi().upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )


def upload_folder(local_dir: str, repo_id: str, path_in_repo: str):
    HfApi().upload_folder(
        folder_path=local_dir,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
    )


def download_folder(repo_id: str, path_in_repo: str, local_dir: str):
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{path_in_repo}/*"],
        local_dir=local_dir,
    )


# ============================================================
# run()
# ============================================================

_DATASET_CLS = {"bgl": BGL, "tbird": Tbird, "spirit": Spirit,
                "liberty": Liberty}


def run(cfg: Config):
    # --- seed ---
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}, cuda_available={torch.cuda.is_available()}")

    # --- HF ---
    if cfg.push_to_hub:
        ensure_login()
        repo_id = resolve_repo_id(cfg)
        ensure_repo(repo_id)
        print(f"[hf] repo={repo_id}")
    else:
        repo_id = None

    # --- run_tag ---
    if cfg.run_tag is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.run_tag = f"run_{ts}_{cfg.dataset}_tr{cfg.train_ratio}"
    run_path_in_repo = f"runs/{cfg.run_tag}"
    print(f"[run] tag={cfg.run_tag}")

    # --- датасет ---
    data_cls = _DATASET_CLS[cfg.dataset]
    max_lines = math.inf if cfg.max_lines is None else cfg.max_lines
    d = data_cls(
        window_size=cfg.win_size,
        step_size=cfg.step_size,
        max_lines=max_lines,
        use_in_colab=cfg.use_in_colab,
    )
    ds = SuperComputerDataset(d, cfg.anchor_step,
                              train_ratio=cfg.eval_start_ratio)
    print(f"[data] total_lines={ds.total_lines}")

    # eval всегда стартует с eval_start_ratio (фиксировано через train_ratio
    # внутри SuperComputerDataset), а тренировка ограничивается train_ratio.
    val_ds = EvaluationDataset(ds)
    eval_start_line = val_ds.start_line
    train_limit = int(ds.total_lines * cfg.train_ratio)
    print(f"[data] eval_start_line={eval_start_line}, "
          f"train_limit={train_limit}")

    # --- кэш эмбеддингов + collate ---
    raw_encoder = load_jasper_encoder(
        cfg.jasper_model_name, cfg.compression_ratio, device=str(device)
    )
    # определяем размерность
    probe = raw_encoder(["probe"])
    jasper_dim = probe.shape[-1]
    del probe
    print(f"[jasper] dim={jasper_dim}")

    line_cache = LineEmbeddingCache(
        raw_encoder, max_lines=cfg.cache_max_lines,
        encode_batch=cfg.jasper_batch,
    )
    collate = make_collate_fn(line_cache)

    # --- сэмплер и лоадеры ---
    base_sampler = BalancedSampler(
        ds, target_ratio=cfg.target_ratio, seed=cfg.seed,
    )
    train_sampler = FilteredSampler(
        base_sampler, max_start=min(train_limit, eval_start_line),
    )
    train_loader = DataLoader(
        ds, batch_size=cfg.batch_size, sampler=train_sampler,
        collate_fn=collate, num_workers=cfg.num_workers,
    )
    val_loader_small = DataLoader(
        LimitedIterable(val_ds, cfg.eval_max_windows),
        batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate, num_workers=cfg.num_workers,
    )
    val_loader_full = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate, num_workers=cfg.num_workers,
    )

    # --- модель ---
    model = LogAnomalyModel(cfg, jasper_dim, device)
    model = model.to(device)
    if not cfg.use_quantization and resolve_dtype(cfg) != torch.float32:
        model = model.to(resolve_dtype(cfg))
    # эмбеддинги Jasper приходят в bf16 — если модель в fp32, это несовместимо.
    if not cfg.use_quantization and resolve_dtype(cfg) == torch.float32:
        # оставляем модель в fp32, будем кастовать в collate/forward
        pass

    # --- resume ---
    start_epoch, global_step = 0, 0
    if cfg.resume_from_repo and cfg.resume_from_run_tag:
        resume_path_in_repo = f"runs/{cfg.resume_from_run_tag}"
        local_ckpt = "/content/ckpt_resume"
        print(f"[resume] download {cfg.resume_from_repo}/{resume_path_in_repo}")
        download_folder(cfg.resume_from_repo, resume_path_in_repo, local_ckpt)
        adapter_dir = os.path.join(
            local_ckpt, resume_path_in_repo.split("/")[-1], "adapter"
        )
        train_state_dir = os.path.join(
            local_ckpt, resume_path_in_repo.split("/")[-1]
        )
        load_adapter_into(model, adapter_dir)

        # optimizer/scheduler создаём до load
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )
        total_steps = max(1, len(train_sampler) // cfg.batch_size * cfg.num_epochs)
        if cfg.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps
            )
        else:
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.warmup_steps,
                num_training_steps=total_steps,
            )
        start_epoch, global_step = load_trainable(
            model, optimizer, scheduler, train_state_dir, device
        )
        print(f"[resume] start_epoch={start_epoch} step={global_step}")
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )
        total_steps = max(1, len(train_sampler) // cfg.batch_size * cfg.num_epochs)
        if cfg.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps
            )
        else:
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=cfg.warmup_steps,
                num_training_steps=total_steps,
            )

    criterion = torch.nn.BCEWithLogitsLoss()
    autocast_ctx = autocast_context(cfg)

    # --- загружаем историю метрик, если она уже есть в репо ---
    local_metrics_path = f"/content/metrics_{cfg.run_tag}.pkl"
    history: list = []
    if repo_id:
        try:
            cached = hf_hub_download(
                repo_id, f"{run_path_in_repo}/metrics.pkl"
            )
            with open(cached, "rb") as f:
                history = pickle.load(f)
            print(f"[metrics] загружено {len(history)} записей")
        except Exception:
            pass

    def append_metric(record: dict):
        history.append(record)
        with open(local_metrics_path, "wb") as f:
            pickle.dump(history, f)
        if repo_id:
            upload_file(local_metrics_path, repo_id,
                        f"{run_path_in_repo}/metrics.pkl")

    # --- конфиг в репо ---
    if repo_id:
        with open("/content/config_run.json", "w") as f:
            f.write(cfg_to_json(cfg))
        upload_file("/content/config_run.json", repo_id,
                    f"{run_path_in_repo}/config.json")

    # --- обучение ---
    epochs_bar = tqdm(range(start_epoch, cfg.num_epochs), desc="training")
    for epoch in epochs_bar:
        model.train()
        running, n_run = 0.0, 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for step, batch in enumerate(pbar):
            if batch is None:
                continue
            log_embs = batch["log_embs"].to(device)
            mask     = batch["attention_mask"].to(device)
            y        = batch["labels"].to(device)

            optimizer.zero_grad()
            with autocast_ctx:
                logits = model(log_embs, batch["times"], mask)
            loss = criterion(logits.float(), y)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip,
                )
            optimizer.step()
            scheduler.step()

            running += loss.item()
            n_run += 1
            global_step += 1

            pbar.set_postfix({
                "loss": f"{running / max(n_run, 1):.3f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })
            # if cfg.max_train_batches is not None and step >= cfg.max_train_batches:
            #     break

            if cfg.metrics_log_every and (step + 1) % cfg.metrics_log_every == 0:
                ev = evaluate(model, val_loader_small, device, criterion,
                              autocast_ctx=autocast_ctx,
                              max_windows=cfg.eval_max_windows)
                rec = {
                    "run_tag": cfg.run_tag,
                    "epoch": epoch,
                    "global_step": global_step,
                    "phase": "intra_epoch",
                    "train_loss": running / max(n_run, 1),
                    "eval": ev,
                    "train_ratio": cfg.train_ratio,
                    "eval_start_ratio": cfg.eval_start_ratio,
                    "elapsed_sec": time.time() - t0,
                }
                append_metric(rec)
                print(f"[e{epoch} s{global_step}] train={running/max(n_run,1):.4f} "
                      f"eval_f1={ev['f1']:.4f} P={ev['precision']:.3f} "
                      f"R={ev['recall']:.3f} n={ev['n_windows']}")
                model.train()

        # --- eval в конце эпохи (мелкий) ---
        ev = evaluate(model, val_loader_small, device, criterion,
                      autocast_ctx=autocast_ctx,
                      max_windows=cfg.eval_max_windows)
        rec = {
            "run_tag": cfg.run_tag,
            "epoch": epoch,
            "global_step": global_step,
            "phase": "end_of_epoch",
            "train_loss": running / max(n_run, 1),
            "eval": ev,
            "train_ratio": cfg.train_ratio,
            "eval_start_ratio": cfg.eval_start_ratio,
            "elapsed_sec": time.time() - t0,
        }
        append_metric(rec)
        print(f"[end e{epoch}] eval_f1={ev['f1']:.4f} "
              f"P={ev['precision']:.3f} R={ev['recall']:.3f} "
              f"n={ev['n_windows']} cache: {line_cache.stats()}")

        # --- чекпоинт на HF ---
        local_ckpt = f"/content/ckpt_{cfg.run_tag}_e{epoch}"
        save_trainable(model, optimizer, scheduler, epoch + 1,
                       global_step, local_ckpt)
        if repo_id:
            upload_folder(local_ckpt, repo_id,
                          f"{run_path_in_repo}/epoch_{epoch:02d}")
            upload_folder(os.path.join(local_ckpt, "adapter"), repo_id,
                          f"{run_path_in_repo}/adapter")   # перезаписываем latest
            upload_file(os.path.join(local_ckpt, "train_state.pt"),
                        repo_id, f"{run_path_in_repo}/train_state.pt")
        print(f"[save] epoch {epoch} → {local_ckpt}")

    # --- финальный полный eval ---
    if cfg.final_full_eval:
        ev = evaluate(model, val_loader_full, device, criterion,
                      autocast_ctx=autocast_ctx, max_windows=None)
        rec = {
            "run_tag": cfg.run_tag,
            "epoch": cfg.num_epochs - 1,
            "global_step": global_step,
            "phase": "final_full",
            "eval": ev,
            "train_ratio": cfg.train_ratio,
            "eval_start_ratio": cfg.eval_start_ratio,
        }
        append_metric(rec)
        print(f"[final full eval] f1={ev['f1']:.4f} "
              f"P={ev['precision']:.3f} R={ev['recall']:.3f} "
              f"n={ev['n_windows']} (pos={ev['n_pos']})")

    # --- финальный чекпоинт ---
    local_final = f"/content/final_{cfg.run_tag}"
    save_trainable(model, optimizer, scheduler, cfg.num_epochs,
                   global_step, local_final)
    if repo_id:
        upload_folder(local_final, repo_id,
                      f"{run_path_in_repo}/final")
        upload_file(local_metrics_path, repo_id,
                    f"{run_path_in_repo}/metrics.pkl")
        # объединённая история по всем run'ам
        merged = "/content/metrics_history.pkl"
        merged_data = []
        if repo_id:
            try:
                cached = hf_hub_download(repo_id, "metrics_history.pkl")
                with open(cached, "rb") as f:
                    merged_data = pickle.load(f)
            except Exception:
                pass
        merged_data.extend(history)
        with open(merged, "wb") as f:
            pickle.dump(merged_data, f)
        upload_file(merged, repo_id, "metrics_history.pkl")
        print(f"[hf] done → https://huggingface.co/{repo_id}")

    return model

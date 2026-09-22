"""
Config + run() + continue_training().
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

import tqdm.auto as tqdm
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
    save_trainable, load_adapter_into,
    resolve_dtype,
)


# ============================================================
# Config
# ============================================================

@dataclasses.dataclass
class Config:
    # данные
    dataset: str = "bgl"
    win_size: int = 199
    step_size: int = 199
    anchor_step: int = 10
    batch_size: int = 16
    num_workers: int = 0
    use_in_colab: bool = True
    max_lines: typing.Optional[int] = None

    # доли
    eval_start_ratio: float = 0.9
    train_ratio: float = 0.3
    target_ratio: float = 0.3

    # eval
    eval_max_windows: int = 2000
    metrics_log_every: int = 100
    final_full_eval: bool = True

    # временные признаки
    time_emb_len: int = 16
    use_time2vec: bool = True
    use_cyclic_time: bool = True
    use_time_present: bool = True

    # Jasper
    jasper_model_name: str = "infgrad/Jasper-Token-Compression-600M"
    compression_ratio: float = 0.5
    jasper_batch: int = 64
    cache_max_lines: int = 50_000

    # Longformer
    longformer_model_name: str = "allenai/longformer-base-4096"
    longformer_layers: int = 12
    longformer_attention_window: int = 100

    # агрегация
    aggregate_layers: int = 4
    aggregate_mode: str = "weighted"
    use_learned_cls: bool = True

    # квантизация
    use_quantization: bool = False
    bnb_4bit: bool = True
    bnb_quant_type: str = "nf4"
    bnb_double_quant: bool = True
    bnb_compute_dtype: str = "bfloat16"
    mixed_dtype: str = "bfloat16"
    use_autocast: bool = False

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.15
    lora_target_modules: typing.List[str] = dataclasses.field(
        default_factory=lambda: ["query", "key", "value", "dense"]
    )

    # обучение
    num_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.05
    warmup_steps: int = 200
    scheduler_type: str = "linear"
    grad_clip: float = 1.0
    seed: int = 42

    # sampler
    sampler_max_oversample: float = 10.0
    sampler_min_minority: int = 50

    # Hub
    push_to_hub: bool = True
    hub_repo_id: typing.Optional[str] = None
    run_tag: typing.Optional[str] = None
    resume_from_repo: typing.Optional[str] = None
    resume_from_run_tag: typing.Optional[str] = None


# ============================================================
# Config ↔ JSON
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
# Samplers
# ============================================================

class FilteredSampler(Sampler):
    """Отсекает индексы >= max_start. Честная __len__."""
    def __init__(self, base: Sampler, max_start: int):
        self.base = base
        self.max_start = max_start
        self._indices: typing.Optional[np.ndarray] = None

    def _materialize(self):
        if self._indices is None:
            self._indices = np.asarray(
                [int(i) for i in self.base if int(i) < self.max_start],
                dtype=np.int64,
            )

    def __iter__(self):
        self._materialize()
        yield from self._indices.tolist()

    def __len__(self):
        self._materialize()
        return int(len(self._indices))


class LimitedIterable(torch.utils.data.IterableDataset):
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
# HF
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
        allow_patterns=[f"{path_in_repo}/**"],
        local_dir=local_dir,
    )


# ============================================================
# train_loop
# ============================================================

def train_loop(model, optimizer, scheduler, train_loader,
               val_loader_small, val_loader_full, criterion, cfg, device,
               autocast_ctx, line_cache, append_metric, history,
               start_epoch: int, global_step: int,
               repo_id: typing.Optional[str]):
    """Возвращает (new_start_epoch, global_step)."""
    end_epoch = start_epoch + cfg.num_epochs
    print(f"[train_loop] epochs {start_epoch}..{end_epoch-1} "
          f"(global_step={global_step})")

    epochs_bar = tqdm.trange(start_epoch, end_epoch, desc="training")
    for epoch in epochs_bar:
        model.train()
        running, n_run = 0.0, 0
        t0 = time.time()

        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
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

            # защита от NaN/Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[skip] NaN/Inf loss на шаге {global_step}")
                optimizer.zero_grad()
                continue

            loss.backward()

            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip,
                )

            has_bad_grad = any(
                p.grad is not None and
                (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in model.parameters() if p.requires_grad
            )
            if has_bad_grad:
                print(f"[skip] NaN/Inf grad на шаге {global_step}")
                optimizer.zero_grad()
                continue

            optimizer.step()
            scheduler.step()

            running += loss.item()
            n_run += 1
            global_step += 1

            pbar.set_postfix({
                "loss": f"{running / max(n_run, 1):.3f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

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
                print(f"[e{epoch} s{global_step}] "
                      f"train={running/max(n_run,1):.4f} "
                      f"eval_f1={ev['f1']:.4f} P={ev['precision']:.3f} "
                      f"R={ev['recall']:.3f} n={ev['n_windows']}")
                model.train()

        # eval в конце эпохи
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

        # чекпоинт
        local_ckpt = f"/content/ckpt_{cfg.run_tag}_e{epoch}"
        save_trainable(model, optimizer, scheduler, epoch + 1,
                       global_step, local_ckpt)
        if repo_id:
            upload_folder(local_ckpt, repo_id,
                          f"runs/{cfg.run_tag}/epoch_{epoch:02d}")
            upload_folder(os.path.join(local_ckpt, "adapter"), repo_id,
                          f"runs/{cfg.run_tag}/adapter")
            upload_file(os.path.join(local_ckpt, "train_state.pt"),
                        repo_id, f"runs/{cfg.run_tag}/train_state.pt")
        print(f"[save] epoch {epoch} → {local_ckpt}")

    # финальный полный eval
    if cfg.final_full_eval:
        ev = evaluate(model, val_loader_full, device, criterion,
                      autocast_ctx=autocast_ctx, max_windows=None,
                      desc="final full eval")
        rec = {
            "run_tag": cfg.run_tag,
            "epoch": end_epoch - 1,
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

    return end_epoch, global_step


# ============================================================
# Вспомогательные сборки
# ============================================================

_DATASET_CLS = {"bgl": BGL, "tbird": Tbird, "spirit": Spirit,
                "liberty": Liberty}


def _build_loaders(ds, val_ds, cfg, line_cache):
    collate = make_collate_fn(line_cache)
    val_loader_small = DataLoader(
        LimitedIterable(val_ds, cfg.eval_max_windows),
        batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate, num_workers=cfg.num_workers,
    )
    val_loader_full = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate, num_workers=cfg.num_workers,
    )
    return val_loader_small, val_loader_full


def _build_train_loader(ds, eval_start_line, cfg, line_cache,
                        base_sampler=None):
    """
    Если base_sampler передан — переиспользуем (экономит минуты).
    Возвращает (train_loader, base_sampler).
    """
    train_limit = int(ds.total_lines * cfg.train_ratio)

    if base_sampler is None:
        print("[sampler] строим BalancedSampler (может занять минуты)...")
        base_sampler = BalancedSampler(
            ds, target_ratio=cfg.target_ratio, seed=cfg.seed,
            max_oversample_factor=getattr(cfg, "sampler_max_oversample", 10.0),
            min_minority_for_balance=getattr(cfg, "sampler_min_minority", 50),
        )
        print("[sampler] готов")
    else:
        print("[sampler] переиспользуем существующий BalancedSampler")

    safe_max = max(0, min(train_limit, eval_start_line - cfg.win_size))
    train_sampler = FilteredSampler(base_sampler, max_start=safe_max)
    collate = make_collate_fn(line_cache)
    train_loader = DataLoader(
        ds, batch_size=cfg.batch_size, sampler=train_sampler,
        collate_fn=collate, num_workers=cfg.num_workers,
    )
    print(f"[loader] train_ratio={cfg.train_ratio} → "
          f"train_limit={train_limit} safe_max={safe_max} "
          f"окон={len(train_sampler)}")
    return train_loader, base_sampler


def _build_scheduler(optimizer, total_steps, cfg):
    if cfg.scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )
    from transformers import get_linear_schedule_with_warmup
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )


def _build_model(cfg, jasper_dim, device):
    model = LogAnomalyModel(cfg, jasper_dim, device)
    if cfg.use_quantization:
        model.projector = model.projector.to(device)
        model.aggregator = model.aggregator.to(device)
        model.classifier = model.classifier.to(device)
        if model.time2vec is not None:
            model.time2vec = model.time2vec.to(device)
        if model.use_cls:
            model.cls_emb = torch.nn.Parameter(model.cls_emb.data.to(device))
    elif cfg.use_autocast:
        model = model.to(device).float()
    else:
        model = model.to(device).to(resolve_dtype(cfg))
    return model


# ============================================================
# run()
# ============================================================

def run(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}, cuda_available={torch.cuda.is_available()}")

    if cfg.push_to_hub:
        ensure_login()
        repo_id = resolve_repo_id(cfg)
        ensure_repo(repo_id)
        print(f"[hf] repo={repo_id}")
    else:
        repo_id = None

    if cfg.run_tag is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.run_tag = f"run_{ts}_{cfg.dataset}_tr{cfg.train_ratio}"
    run_path_in_repo = f"runs/{cfg.run_tag}"
    print(f"[run] tag={cfg.run_tag}")

    # датасет
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

    val_ds = EvaluationDataset(ds)
    eval_start_line = val_ds.start_line
    print(f"[data] eval_start_line={eval_start_line}")

    # Jasper + кэш
    raw_encoder = load_jasper_encoder(
        cfg.jasper_model_name, cfg.compression_ratio, device=str(device)
    )
    probe = raw_encoder(["probe"])
    jasper_dim = probe.shape[-1]
    del probe
    print(f"[jasper] dim={jasper_dim}")

    line_cache = LineEmbeddingCache(
        raw_encoder, max_lines=cfg.cache_max_lines,
        encode_batch=cfg.jasper_batch,
    )

    # лоадеры
    train_loader, base_sampler = _build_train_loader(
        ds, eval_start_line, cfg, line_cache
    )
    val_loader_small, val_loader_full = _build_loaders(
        ds, val_ds, cfg, line_cache
    )

    # модель
    model = _build_model(cfg, jasper_dim, device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[opt] trainable={n_train:,} / total={n_total:,} "
          f"({n_train / max(n_total, 1) * 100:.2f}%)")

    # resume
    start_epoch, global_step = 0, 0
    ckpt_state = None

    if cfg.resume_from_repo and cfg.resume_from_run_tag:
        resume_path_in_repo = f"runs/{cfg.resume_from_run_tag}"
        local_ckpt = "/content/ckpt_resume"
        print(f"[resume] download {cfg.resume_from_repo}/{resume_path_in_repo}")
        download_folder(cfg.resume_from_repo, resume_path_in_repo, local_ckpt)

        adapter_dir = os.path.join(local_ckpt, resume_path_in_repo, "adapter")
        train_state_dir = os.path.join(local_ckpt, resume_path_in_repo)

        print(f"[resume] adapter_dir={adapter_dir}")
        print(f"[resume] train_state_dir={train_state_dir}")
        assert os.path.isdir(adapter_dir), f"нет папки адаптера: {adapter_dir}"
        assert os.path.isfile(os.path.join(train_state_dir, "train_state.pt")), \
            f"нет train_state.pt в {train_state_dir}"

        load_adapter_into(model, adapter_dir)

        ckpt_state = torch.load(
            os.path.join(train_state_dir, "train_state.pt"),
            map_location="cpu",
        )
        model.load_head_state_dict(ckpt_state["head_state"])
        start_epoch = ckpt_state.get("epoch", 0)
        global_step = ckpt_state.get("global_step", 0)
        print(f"[resume] start_epoch={start_epoch} step={global_step}")

    # optimizer + scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )
    total_steps = max(1, len(train_loader) * cfg.num_epochs)
    scheduler = _build_scheduler(optimizer, total_steps, cfg)

    if ckpt_state is not None:
        try:
            optimizer.load_state_dict(ckpt_state["optimizer"])
            print("[resume] optimizer state восстановлен")
        except Exception as e:
            print(f"[resume] optimizer не восстановился: {e}")
        try:
            scheduler.load_state_dict(ckpt_state["scheduler"])
            print("[resume] scheduler state восстановлен")
        except Exception as e:
            print(f"[resume] scheduler не восстановился: {e}")

    criterion = torch.nn.BCEWithLogitsLoss()
    autocast_ctx = autocast_context(cfg)

    # история метрик
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

    # конфиг в репо
    if repo_id:
        with open("/content/config_run.json", "w") as f:
            f.write(cfg_to_json(cfg))
        upload_file("/content/config_run.json", repo_id,
                    f"{run_path_in_repo}/config.json")

    # обучение
    start_epoch, global_step = train_loop(
        model, optimizer, scheduler, train_loader,
        val_loader_small, val_loader_full, criterion, cfg, device,
        autocast_ctx, line_cache, append_metric, history,
        start_epoch, global_step, repo_id,
    )

    # финальный чекпоинт
    local_final = f"/content/final_{cfg.run_tag}"
    save_trainable(model, optimizer, scheduler, start_epoch,
                   global_step, local_final)
    if repo_id:
        upload_folder(local_final, repo_id,
                      f"{run_path_in_repo}/final")
        upload_file(local_metrics_path, repo_id,
                    f"{run_path_in_repo}/metrics.pkl")

        # объединённая история с дедупликацией
        merged = "/content/metrics_history.pkl"
        merged_data = []
        try:
            cached = hf_hub_download(repo_id, "metrics_history.pkl")
            with open(cached, "rb") as f:
                merged_data = pickle.load(f)
        except Exception:
            pass

        def _key(r):
            return (r.get("run_tag"), r.get("global_step"), r.get("phase"))

        seen = {_key(r) for r in merged_data}
        new_records = [r for r in history if _key(r) not in seen]
        print(f"[metrics_history] было {len(merged_data)}, "
              f"новых {len(new_records)}, "
              f"дубликатов пропущено {len(history) - len(new_records)}")

        merged_data.extend(new_records)
        with open(merged, "wb") as f:
            pickle.dump(merged_data, f)
        upload_file(merged, repo_id, "metrics_history.pkl")
        print(f"[hf] done → https://huggingface.co/{repo_id}")

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "val_loader_small": val_loader_small,
        "val_loader_full": val_loader_full,
        "criterion": criterion,
        "autocast_ctx": autocast_ctx,
        "line_cache": line_cache,
        "history": history,
        "append_metric": append_metric,
        "global_step": global_step,
        "start_epoch": start_epoch,
        "dataset": ds,
        "val_dataset": val_ds,
        "eval_start_line": eval_start_line,
        "train_ratio": cfg.train_ratio,
        "run_tag": cfg.run_tag,
        "cfg": cfg,
        "device": device,
        "repo_id": repo_id,
        "base_sampler": base_sampler,        # ← сохраняем для переиспользования
    }


# ============================================================
# continue_training
# ============================================================

def continue_training(state: dict, cfg: Config) -> dict:
    """In-memory продолжение. Пересобирает train_loader при смене ratio."""
    if cfg.warmup_steps > 0:
        print(f"[continue] warn: warmup_steps={cfg.warmup_steps} > 0. "
              f"Для дообучения рекомендуется warmup_steps=0.")

    model = state["model"]
    optimizer = state["optimizer"]
    criterion = state["criterion"]
    device = state["device"]
    line_cache = state["line_cache"]
    history = state["history"]
    global_step = state["global_step"]
    autocast_ctx = state["autocast_ctx"]
    ds = state["dataset"]
    val_ds = state["val_dataset"]
    eval_start_line = state["eval_start_line"]

    # resolve repo_id
    if cfg.push_to_hub:
        repo_id = (cfg.hub_repo_id
                   or state.get("repo_id")
                   or resolve_repo_id(cfg))
    else:
        repo_id = None

    # train_loader
    if cfg.train_ratio != state.get("train_ratio") \
       or cfg.batch_size != state["cfg"].batch_size:
        train_loader, base_sampler = _build_train_loader(
            ds, eval_start_line, cfg, line_cache,
            base_sampler=state.get("base_sampler"),
        )
        state["base_sampler"] = base_sampler
    else:
        train_loader = state["train_loader"]

    # val_loader при смене batch_size
    if cfg.batch_size != state["cfg"].batch_size:
        val_loader_small, val_loader_full = _build_loaders(
            ds, val_ds, cfg, line_cache
        )
    else:
        val_loader_small = state["val_loader_small"]
        val_loader_full = state["val_loader_full"]

    # scheduler
    total_steps = max(1, len(train_loader) * cfg.num_epochs)
    scheduler = _build_scheduler(optimizer, total_steps, cfg)

    # append_metric
    local_metrics_path = f"/content/metrics_{cfg.run_tag}.pkl"

    def append_metric(record: dict):
        record["run_tag"] = cfg.run_tag
        history.append(record)
        with open(local_metrics_path, "wb") as f:
            pickle.dump(history, f)
        if repo_id:
            upload_file(local_metrics_path, repo_id,
                        f"runs/{cfg.run_tag}/metrics.pkl")

    # конфиг
    if repo_id:
        with open("/content/config_run.json", "w") as f:
            f.write(cfg_to_json(cfg))
        upload_file("/content/config_run.json", repo_id,
                    f"runs/{cfg.run_tag}/config.json")

    # train_loop
    start_epoch = state["start_epoch"]
    start_epoch, global_step = train_loop(
        model, optimizer, scheduler, train_loader,
        val_loader_small, val_loader_full, criterion, cfg, device,
        autocast_ctx, line_cache, append_metric, history,
        start_epoch, global_step, repo_id,
    )

    # финальный чекпоинт
    local_final = f"/content/final_{cfg.run_tag}"
    save_trainable(model, optimizer, scheduler, start_epoch,
                   global_step, local_final)
    if repo_id:
        upload_folder(local_final, repo_id,
                      f"runs/{cfg.run_tag}/final")
        upload_file(local_metrics_path, repo_id,
                    f"runs/{cfg.run_tag}/metrics.pkl")

    state.update({
        "cfg": cfg,
        "train_loader": train_loader,
        "val_loader_small": val_loader_small,
        "val_loader_full": val_loader_full,
        "scheduler": scheduler,
        "run_tag": cfg.run_tag,
        "train_ratio": cfg.train_ratio,
        "start_epoch": start_epoch,
        "global_step": global_step,
        "append_metric": append_metric,
        "repo_id": repo_id,
    })
    return state
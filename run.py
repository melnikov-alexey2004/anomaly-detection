"""
Config + run() + continue_training().
"""
import os
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

from peft import PeftConfig
from transformers import BitsAndBytesConfig

from dataset import (
    BGL, Tbird, Liberty, Spirit,
    SuperComputerDataset, EvaluationDataset, BalancedSampler,
)
from model import (
    load_jasper_encoder, LineEmbeddingCache, make_collate_fn,
    LogAnomalyModel, EncoderSpec,
    evaluate, collect_predictions,
    save_trainable, load_adapter_into,
    resolve_dtype,
)


# ============================================================
# Config
# ============================================================

@dataclasses.dataclass
class Config:
    # ---------- данные ----------
    dataset: str = "bgl"
    win_size: int = 199
    step_size: int = 199
    anchor_step: int = 10
    batch_size: int = 16
    num_workers: int = 0
    use_in_colab: bool = True
    max_lines: typing.Optional[int] = None
    start_line: int = 0
    end_line: typing.Optional[int] = None

    # ---------- доли ----------
    eval_start_ratio: float = 0.9
    train_ratio: float = 0.3
    target_ratio: float = 0.3

    # ---------- eval ----------
    eval_max_windows: typing.Optional[int] = 2000
    metrics_log_every: int = 100
    final_full_eval: bool = True

    # ---------- временные признаки ----------
    time_emb_len: int = 16
    use_time2vec: bool = True
    use_cyclic_time: bool = True
    use_time_present: bool = True

    # ---------- Jasper ----------
    jasper_model_name: str = "infgrad/Jasper-Token-Compression-600M"
    compression_ratio: float = 0.5
    jasper_batch: int = 64
    cache_max_lines: int = 50_000

    # ---------- Longformer (используется _build_longformer_encoder) ----------
    longformer_model_name: str = "allenai/longformer-base-4096"
    longformer_layers: int = 12
    longformer_attention_window: int = 100

    # ---------- выбор энкодера / модели ----------
    encoder_type: str = "longformer"
    encoder_factory: typing.Optional[typing.Callable] = None
    model_factory:   typing.Optional[typing.Callable] = None

    # ---------- агрегация ----------
    aggregate_layers: int = 4
    aggregate_mode: str = "weighted"
    use_learned_cls: bool = True

    # ---------- dtype весов / квантизация / FP8 ----------
    # "bf16" | "fp16" | "fp32" | "tf32"
    model_dtype: str = "bf16"
    # None → обычный float dtype; иначе — BitsAndBytesConfig (QLoRA)
    quantization_config: typing.Optional[BitsAndBytesConfig] = None
    # FP8 (Ada/Hopper/Blackwell). Несовместим с quantization_config.
    fp8_training: bool = False
    fp8_backend: str = "torchao"      # "torchao" | "transformer_engine"
    fp8_recipe: str = "rowwise"       # "tensorwise" | "rowwise" | "rowwise_with_gw_hp"

    # ---------- PEFT ----------
    # Единственный источник истины: готовый PeftConfig. Без fallback.
    peft_config: typing.Optional[PeftConfig] = None

    # ---------- обучение ----------
    num_epochs: int = 3
    learning_rate: float = 5e-5
    lr_end: float = 1e-7
    weight_decay: float = 0.05
    warmup_steps: int = 200
    scheduler_type: str = "linear"
    grad_clip: float = 1.0
    seed: int = 42

    # ---------- sampler ----------
    sampler_max_oversample: float = 10.0
    sampler_min_minority: int = 50
    max_samples: typing.Optional[int] = None
    min_samples: typing.Optional[int] = None

    # ---------- режим тренировки ----------
    train_mode: str = "all"
    mixed_prev_fraction: float = 0.1

    # ---------- сохранение предсказаний ----------
    save_predictions: bool = True
    predictions_dir: str = "/content/predictions"

    # ---------- Hub ----------
    push_to_hub: bool = True
    hub_repo_id: typing.Optional[str] = None
    run_tag: typing.Optional[str] = None
    resume_from_repo: typing.Optional[str] = None
    resume_from_run_tag: typing.Optional[str] = None


# ============================================================
# Config <-> JSON
# ============================================================

def cfg_to_json_dict(cfg: Config) -> dict:
    d = dataclasses.asdict(cfg)
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.dtype):
            out[k] = str(v)
        elif k == "quantization_config":
            if v is None:
                out[k] = None
            else:
                out[k] = {"_type": type(v).__name__,
                          **{kk: str(vv) for kk, vv in v.to_dict().items()}}
        elif k == "peft_config":
            if v is None:
                out[k] = None
            else:
                out[k] = {"_type": type(v).__name__,
                          **{kk: str(vv) for kk, vv in v.to_dict().items()}}
        elif k in ("encoder_factory", "model_factory"):
            out[k] = None if v is None else f"<callable {getattr(v, '__name__', v)}>"
        else:
            out[k] = v
    return out


def cfg_to_json(cfg: Config) -> str:
    return json.dumps(cfg_to_json_dict(cfg), indent=2, ensure_ascii=False)


# ============================================================
# Samplers
# ============================================================

class FilteredSampler(Sampler):
    def __init__(self, base, max_start: int, min_start: int = 0):
        self.base = base
        self.max_start = max_start
        self.min_start = min_start
        self._indices = None

    def _materialize(self):
        if self._indices is None:
            self._indices = np.asarray(
                [int(i) for i in self.base
                 if self.min_start <= int(i) < self.max_start],
                dtype=np.int64)

    def __iter__(self):
        self._materialize()
        yield from self._indices.tolist()

    def __len__(self):
        self._materialize()
        return int(len(self._indices))


class LimitedIterable(torch.utils.data.IterableDataset):
    def __init__(self, base, max_items):
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


class RangeSampler(Sampler):
    """Диапазон [start, end) с шагом step, без oversample."""
    def __init__(self, start: int, end: int, step: int):
        self.indices = list(range(start, end, step))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


# ============================================================
# HF
# ============================================================

def ensure_login():
    try:
        whoami()
    except Exception:
        notebook_login()


def resolve_repo_id(cfg) -> str:
    if cfg.hub_repo_id:
        return cfg.hub_repo_id
    user = whoami()["name"]
    return f"{user}/{cfg.dataset}-log-anomaly"


def ensure_repo(repo_id: str):
    HfApi().create_repo(repo_id, repo_type="model", exist_ok=True)


def upload_file(local_path, repo_id, path_in_repo):
    HfApi().upload_file(path_or_fileobj=local_path,
                        path_in_repo=path_in_repo,
                        repo_id=repo_id, repo_type="model")


def upload_folder(local_dir, repo_id, path_in_repo):
    HfApi().upload_folder(folder_path=local_dir,
                          path_in_repo=path_in_repo,
                          repo_id=repo_id, repo_type="model")


def download_folder(repo_id, path_in_repo, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="model",
                      allow_patterns=[f"{path_in_repo}/**"],
                      local_dir=local_dir)


# ============================================================
# train_loop
# ============================================================

def train_loop(model, optimizer, scheduler, train_loader,
               val_loader_small, val_loader_full, criterion, cfg, device,
               line_cache, ds, append_metric, history,
               start_epoch: int, global_step: int,
               repo_id: typing.Optional[str]):
    end_epoch = start_epoch + cfg.num_epochs
    print(f"[train_loop] epochs {start_epoch}..{end_epoch-1} "
          f"(global_step={global_step})")

    epochs_bar = tqdm.trange(start_epoch, end_epoch, desc="training")
    for epoch in epochs_bar:
        model.train()
        running, n_run = 0.0, 0
        t0 = time.time()

        pbar = tqdm.tqdm(train_loader, desc=f"epoch {epoch}", leave=True)
        for step, batch in enumerate(pbar):
            if batch is None:
                continue
            log_embs = batch["log_embs"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(log_embs, batch["times"], mask)
            loss = criterion(logits.float(), y)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[skip] NaN/Inf loss at step {global_step}")
                optimizer.zero_grad()
                continue

            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip)

            has_bad_grad = any(
                p.grad is not None and
                (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in model.parameters() if p.requires_grad)
            if has_bad_grad:
                print(f"[skip] NaN/Inf grad at step {global_step}")
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
                    "target_ratio": cfg.target_ratio,
                    "elapsed_sec": time.time() - t0,
                }
                append_metric(rec)
                print(f"[e{epoch} s{global_step}] "
                      f"train={running/max(n_run,1):.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} "
                      f"elapsed={time.time() - t0:.0f}s "
                      f"eval_f1={ev['f1']:.4f} "
                      f"P={ev['precision']:.3f} R={ev['recall']:.3f} "
                      f"n={ev['n_windows']}")
                model.train()

        # eval в конце эпохи
        ev = evaluate(model, val_loader_small, device, criterion,
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
            "target_ratio": cfg.target_ratio,
            "elapsed_sec": time.time() - t0,
        }
        append_metric(rec)
        print(f"[end e{epoch}] "
              f"loss={running / max(n_run, 1):.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e} "
              f"elapsed={time.time() - t0:.0f}s "
              f"F1={ev['f1']:.4f} P={ev['precision']:.3f} "
              f"R={ev['recall']:.3f} n={ev['n_windows']} "
              f"cache: {line_cache.stats()}")

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
        print(f"[save] epoch {epoch} -> {local_ckpt}")

        # предсказания после каждой эпохи
        if cfg.save_predictions:
            _save_predictions(model, ds, train_loader, val_loader_small,
                              cfg, device, line_cache,
                              epoch=epoch, phase="epoch")

    # финальный полный eval
    if cfg.final_full_eval:
        ev = evaluate(model, val_loader_full, device, criterion,
                      max_windows=None, desc="final full eval")
        rec = {
            "run_tag": cfg.run_tag,
            "epoch": end_epoch - 1,
            "global_step": global_step,
            "phase": "final_full",
            "eval": ev,
            "train_ratio": cfg.train_ratio,
            "eval_start_ratio": cfg.eval_start_ratio,
            "target_ratio": cfg.target_ratio,
        }
        append_metric(rec)
        print(f"[final full eval] f1={ev['f1']:.4f} "
              f"P={ev['precision']:.3f} R={ev['recall']:.3f} "
              f"n={ev['n_windows']} (pos={ev['n_pos']})")

        if cfg.save_predictions:
            _save_predictions(model, ds, train_loader, val_loader_full,
                              cfg, device, line_cache,
                              epoch=end_epoch - 1, phase="final")

    return end_epoch, global_step


# ============================================================
# Вспомогательные сборки
# ============================================================

_DATASET_CLS = {"bgl": BGL, "tbird": Tbird, "spirit": Spirit,
                "liberty": Liberty}


def _log_zone_distribution(base_sampler, safe_min, safe_max, name="zone"):
    min_idx = base_sampler.minority_indices
    maj_idx = base_sampler.majority_indices
    if base_sampler.minority_label == "abnormal":
        abnorm_idx, normal_idx = min_idx, maj_idx
    else:
        abnorm_idx, normal_idx = maj_idx, min_idx

    mask_a = (abnorm_idx >= safe_min) & (abnorm_idx < safe_max)
    mask_n = (normal_idx >= safe_min) & (normal_idx < safe_max)
    n_abnorm, n_normal = int(mask_a.sum()), int(mask_n.sum())
    total = n_abnorm + n_normal
    if total == 0:
        print(f"[{name}] зона [{safe_min}, {safe_max}): окон нет")
        return
    print(f"[{name}] зона [{safe_min}, {safe_max}): "
          f"окон={total}, abnormal={n_abnorm} "
          f"({n_abnorm/total*100:.2f}%), normal={n_normal} "
          f"({n_normal/total*100:.2f}%)")


def _build_loaders(ds, val_ds, cfg, line_cache, base_sampler=None):
    collate = make_collate_fn(line_cache)
    val_loader_small = DataLoader(
        LimitedIterable(val_ds, cfg.eval_max_windows),
        batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate, num_workers=cfg.num_workers)
    val_loader_full = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate, num_workers=cfg.num_workers)
    if base_sampler is not None:
        _log_zone_distribution(base_sampler, val_ds.start_line,
                               val_ds.end_line, name="eval_zone")
    return val_loader_small, val_loader_full


def _build_train_loader(ds, eval_start_line, cfg, line_cache,
                        base_sampler=None, prev_train_limit=None):
    train_limit = int(ds.total_lines * cfg.train_ratio)
    safe_max = max(0, min(train_limit, eval_start_line - cfg.win_size))

    if base_sampler is None:
        print("[sampler] строим BalancedSampler (может занять минуты)...")
        base_sampler = BalancedSampler(
            ds, target_ratio=cfg.target_ratio, seed=cfg.seed,
            min_samples=cfg.min_samples, max_samples=cfg.max_samples,
            max_oversample_factor=cfg.sampler_max_oversample,
            min_minority_for_balance=cfg.sampler_min_minority)
        print("[sampler] готов")
    else:
        print("[sampler] переиспользуем существующий BalancedSampler")

    safe_min = 0
    if prev_train_limit is not None and prev_train_limit > 0:
        safe_min = min(prev_train_limit, safe_max)
        print(f"[loader] РЕЖИМ: только новые данные, "
              f"окна в [{safe_min}, {safe_max})")

    train_sampler = FilteredSampler(
        base_sampler, max_start=safe_max, min_start=safe_min)
    collate = make_collate_fn(line_cache)
    train_loader = DataLoader(
        ds, batch_size=cfg.batch_size, sampler=train_sampler,
        collate_fn=collate, num_workers=cfg.num_workers)
    print(f"[loader] train_ratio={cfg.train_ratio} -> "
          f"train_limit={train_limit} safe_min={safe_min} "
          f"safe_max={safe_max} окон={len(train_sampler)}")
    _log_zone_distribution(base_sampler, safe_min, safe_max,
                           name=f"train_zone r={cfg.train_ratio}")
    return train_loader, base_sampler


def _build_scheduler(optimizer, total_steps, cfg):
    if cfg.scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps)
    from transformers import get_polynomial_decay_schedule_with_warmup
    return get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
        lr_end=cfg.lr_end,
        power=1)


def _build_model(cfg, jasper_dim, device):
    if getattr(cfg, "model_factory", None) is not None:
        print("[model] using custom model_factory")
        model = cfg.model_factory(cfg, jasper_dim, device)
    else:
        model = LogAnomalyModel(cfg, jasper_dim, device)

    if cfg.quantization_config is not None:
        # bnb сам управляет device у base-части; головы переносим вручную
        for name in ("projector", "aggregator", "classifier", "time2vec"):
            m = getattr(model, name, None)
            if isinstance(m, nn.Module):
                setattr(model, name, m.to(device))
        if getattr(model, "use_cls", False):
            model.cls_emb = nn.Parameter(model.cls_emb.data.to(device))
    else:
        model = model.to(device)
        if cfg.model_dtype != "fp32" and cfg.model_dtype != "tf32":
            model = model.to(resolve_dtype(cfg))
    return model


def _save_predictions(model, ds, train_loader, val_loader,
                      cfg, device, line_cache,
                      epoch: int, phase: str):
    """
    Сохраняет:
      - eval: полный val_loader (small в конце эпохи, full на final)
      - train: уникальные окна в [safe_min, safe_max) без oversample
    Файлы перезаписываются. На диске всегда последнее состояние.
    """
    os.makedirs(cfg.predictions_dir, exist_ok=True)

    # ---------- eval ----------
    print(f"[pred] collecting eval (phase={phase})...")
    ev = collect_predictions(model, val_loader, device, desc="pred eval")
    eval_path = os.path.join(cfg.predictions_dir,
                             f"pred_{cfg.run_tag}_eval.npz")
    np.savez_compressed(
        eval_path,
        logits=ev["logits"], probs=ev["probs"],
        bce=ev["bce"], labels=ev["labels"],
        epoch=np.int32(epoch), phase=np.array(phase))
    print(f"[pred] saved {eval_path} | n={len(ev['labels'])}, "
          f"pos={int(ev['labels'].sum())}")

    # ---------- train (уникальные окна) ----------
    fs = train_loader.sampler
    safe_min = getattr(fs, "min_start", 0)
    safe_max = getattr(fs, "max_start", ds.total_lines)
    train_sampler = RangeSampler(safe_min, safe_max, cfg.step_size)

    collate = make_collate_fn(line_cache)
    train_unique_loader = DataLoader(
        ds, batch_size=cfg.batch_size, sampler=train_sampler,
        collate_fn=collate, num_workers=cfg.num_workers)
    print(f"[pred] collecting train on {len(train_sampler)} unique windows...")
    tr = collect_predictions(model, train_unique_loader, device,
                             desc="pred train")
    train_path = os.path.join(cfg.predictions_dir,
                              f"pred_{cfg.run_tag}_train.npz")
    starts = np.asarray(train_sampler.indices, dtype=np.int64)
    np.savez_compressed(
        train_path,
        logits=tr["logits"], probs=tr["probs"],
        bce=tr["bce"], labels=tr["labels"],
        starts=starts, epoch=np.int32(epoch), phase=np.array(phase))
    print(f"[pred] saved {train_path} | n={len(tr['labels'])}, "
          f"pos={int(tr['labels'].sum())}")


# ============================================================
# run()
# ============================================================

def run(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}, cuda={torch.cuda.is_available()}")

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

    data_cls = _DATASET_CLS[cfg.dataset]
    max_lines = math.inf if cfg.max_lines is None else cfg.max_lines
    d = data_cls(
        window_size=cfg.win_size, step_size=cfg.step_size,
        max_lines=max_lines,
        start_line=cfg.start_line, end_line=cfg.end_line,
        use_in_colab=cfg.use_in_colab)
    ds = SuperComputerDataset(d, cfg.anchor_step,
                              train_ratio=cfg.eval_start_ratio)
    print(f"[data] total_lines={ds.total_lines}")

    val_ds = EvaluationDataset(ds)
    eval_start_line = val_ds.start_line
    print(f"[data] eval_start_line={eval_start_line}")

    raw_encoder = load_jasper_encoder(
        cfg.jasper_model_name, cfg.compression_ratio, device=str(device))
    probe = raw_encoder(["probe"])
    jasper_dim = probe.shape[-1]
    del probe
    print(f"[jasper] dim={jasper_dim}")

    line_cache = LineEmbeddingCache(
        raw_encoder, max_lines=cfg.cache_max_lines,
        encode_batch=cfg.jasper_batch)

    train_loader, base_sampler = _build_train_loader(
        ds, eval_start_line, cfg, line_cache)
    val_loader_small, val_loader_full = _build_loaders(
        ds, val_ds, cfg, line_cache, base_sampler=base_sampler)

    model = _build_model(cfg, jasper_dim, device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[opt] trainable={n_train:,} / total={n_total:,} "
          f"({n_train / max(n_total, 1) * 100:.2f}%)")

    start_epoch, global_step = 0, 0
    ckpt_state = None

    if cfg.resume_from_repo and cfg.resume_from_run_tag:
        resume_path_in_repo = f"runs/{cfg.resume_from_run_tag}"
        local_ckpt = "/content/ckpt_resume"
        print(f"[resume] download {cfg.resume_from_repo}/{resume_path_in_repo}")
        download_folder(cfg.resume_from_repo, resume_path_in_repo, local_ckpt)

        train_state_dir = os.path.join(local_ckpt, resume_path_in_repo)
        print(f"[resume] train_state_dir={train_state_dir}")
        assert os.path.isdir(train_state_dir), \
            f"нет папки: {train_state_dir}"

        load_adapter_into(model, train_state_dir)
        ckpt_state = torch.load(
            os.path.join(train_state_dir, "train_state.pt"),
            map_location="cpu")
        if ckpt_state.get("head_state") is not None:
            model.load_head_state_dict(ckpt_state["head_state"])
        start_epoch = ckpt_state.get("epoch", 0)
        global_step = ckpt_state.get("global_step", 0)
        print(f"[resume] start_epoch={start_epoch} step={global_step}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.num_epochs)
    scheduler = _build_scheduler(optimizer, total_steps, cfg)

    if ckpt_state is not None:
        try:
            optimizer.load_state_dict(ckpt_state["optimizer"])
            print("[resume] optimizer state восстановлен")
        except Exception as e:
            print(f"[resume] optimizer: {e}")
        try:
            scheduler.load_state_dict(ckpt_state["scheduler"])
            print("[resume] scheduler state восстановлен")
        except Exception as e:
            print(f"[resume] scheduler: {e}")

    criterion = torch.nn.BCEWithLogitsLoss()

    local_metrics_path = f"/content/metrics_{cfg.run_tag}.pkl"
    history: list = []
    if repo_id:
        try:
            cached = hf_hub_download(
                repo_id, f"{run_path_in_repo}/metrics.pkl")
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

    if repo_id:
        with open("/content/config_run.json", "w") as f:
            f.write(cfg_to_json(cfg))
        upload_file("/content/config_run.json", repo_id,
                    f"{run_path_in_repo}/config.json")

    start_epoch, global_step = train_loop(
        model, optimizer, scheduler, train_loader,
        val_loader_small, val_loader_full, criterion, cfg, device,
        line_cache, ds, append_metric, history,
        start_epoch, global_step, repo_id)

    local_final = f"/content/final_{cfg.run_tag}"
    save_trainable(model, optimizer, scheduler, start_epoch,
                   global_step, local_final)
    if repo_id:
        upload_folder(local_final, repo_id, f"{run_path_in_repo}/final")
        upload_file(local_metrics_path, repo_id,
                    f"{run_path_in_repo}/metrics.pkl")

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
        merged_data.extend(new_records)
        with open(merged, "wb") as f:
            pickle.dump(merged_data, f)
        upload_file(merged, repo_id, "metrics_history.pkl")
        print(f"[hf] done -> https://huggingface.co/{repo_id}")

    return {
        "model": model, "optimizer": optimizer, "scheduler": scheduler,
        "train_loader": train_loader,
        "val_loader_small": val_loader_small,
        "val_loader_full": val_loader_full,
        "criterion": criterion,
        "line_cache": line_cache, "history": history,
        "append_metric": append_metric,
        "global_step": global_step, "start_epoch": start_epoch,
        "dataset": ds, "val_dataset": val_ds,
        "eval_start_line": eval_start_line,
        "train_ratio": cfg.train_ratio, "run_tag": cfg.run_tag,
        "cfg": cfg, "device": device, "repo_id": repo_id,
        "base_sampler": base_sampler,
    }


# ============================================================
# continue_training
# ============================================================

def continue_training(state: dict, cfg: Config) -> dict:
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
    ds = state["dataset"]
    val_ds = state["val_dataset"]
    eval_start_line = state["eval_start_line"]

    if cfg.push_to_hub:
        repo_id = (cfg.hub_repo_id or state.get("repo_id")
                   or resolve_repo_id(cfg))
    else:
        repo_id = None

    prev_train_limit = None
    train_mode = getattr(cfg, "train_mode", "all")

    if train_mode != "all" and cfg.train_ratio != state.get("train_ratio"):
        prev_ratio = state.get("train_ratio", 0.0)
        if prev_ratio > 0:
            prev_train_limit = int(ds.total_lines * prev_ratio)
            if train_mode == "mixed":
                frac = getattr(cfg, "mixed_prev_fraction", 0.1)
                prev_train_limit = int(prev_train_limit * (1.0 - frac))
                print(f"[continue] train_mode=mixed, "
                      f"prev_train_limit={prev_train_limit}")
            else:
                print(f"[continue] train_mode=new_only, "
                      f"prev_train_limit={prev_train_limit}")

    base_sampler = state.get("base_sampler")
    if cfg.train_ratio != state.get("train_ratio") \
            or cfg.batch_size != state["cfg"].batch_size:
        train_loader, base_sampler = _build_train_loader(
            ds, eval_start_line, cfg, line_cache,
            base_sampler=base_sampler,
            prev_train_limit=prev_train_limit)
        state["base_sampler"] = base_sampler
    else:
        train_loader = state["train_loader"]

    if cfg.batch_size != state["cfg"].batch_size:
        val_loader_small, val_loader_full = _build_loaders(
            ds, val_ds, cfg, line_cache, base_sampler=base_sampler)
    else:
        val_loader_small = state["val_loader_small"]
        val_loader_full = state["val_loader_full"]

    total_steps = max(1, len(train_loader) * cfg.num_epochs)
    scheduler = _build_scheduler(optimizer, total_steps, cfg)

    local_metrics_path = f"/content/metrics_{cfg.run_tag}.pkl"

    def append_metric(record: dict):
        record["run_tag"] = cfg.run_tag
        history.append(record)
        with open(local_metrics_path, "wb") as f:
            pickle.dump(history, f)
        if repo_id:
            upload_file(local_metrics_path, repo_id,
                        f"runs/{cfg.run_tag}/metrics.pkl")

    if repo_id:
        with open("/content/config_run.json", "w") as f:
            f.write(cfg_to_json(cfg))
        upload_file("/content/config_run.json", repo_id,
                    f"runs/{cfg.run_tag}/config.json")

    start_epoch = state["start_epoch"]
    start_epoch, global_step = train_loop(
        model, optimizer, scheduler, train_loader,
        val_loader_small, val_loader_full, criterion, cfg, device,
        line_cache, ds, append_metric, history,
        start_epoch, global_step, repo_id)

    local_final = f"/content/final_{cfg.run_tag}"
    save_trainable(model, optimizer, scheduler, start_epoch,
                   global_step, local_final)
    if repo_id:
        upload_folder(local_final, repo_id, f"runs/{cfg.run_tag}/final")
        upload_file(local_metrics_path, repo_id,
                    f"runs/{cfg.run_tag}/metrics.pkl")

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
        merged_data.extend(new_records)
        with open(merged, "wb") as f:
            pickle.dump(merged_data, f)
        upload_file(merged, repo_id, "metrics_history.pkl")

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
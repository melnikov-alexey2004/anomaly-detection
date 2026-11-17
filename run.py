"""
LogLLM runner: Config + collator + tests + 4-stage training + continue_learning.

Stages:
  stage1 'llama'   — Llama LoRA only (shared template via HF)
  stage2 'head'    — projector + classifier + aggregator (+ time fusion)
  stage3 'encoder' — head + encoder LoRA
  stage4 'all'     — head + encoder + llama LoRA (несколько эпох)
"""
import os
import math
import time
import random
import typing
import dataclasses
import datetime as _dt

import numpy as np
import torch
import torch.nn as nn
import tqdm.auto as tqdm
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoTokenizer, get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from torch.optim.lr_scheduler import LambdaLR

from dataset import (
    BGL, Tbird, Spirit, Liberty,
    SuperComputerDataset, EvaluationDataset, BalancedSampler,
    get_preprocess_fn, normalize_text,
)
from model import LogLLM, ENCODER_PRESETS, resolve_encoder
from typing import Literal, Optional, Union


# ============================================================
# Config
# ============================================================

def _stub_preprocess(s: str) -> str:
    """Заглушка: ваш старый preprocessing (normalize_text)."""
    return normalize_text(s)


@dataclasses.dataclass
class Config:
    # ---- data ----
    dataset: str = "tbird"
    win_size: int = 100
    step_size: int = 100
    anchor_step: int = 10
    batch_size: int = 4
    num_workers: int = 4
    use_in_colab: bool = True
    max_lines: typing.Optional[int] = None
    start_line: int = 0
    end_line: typing.Optional[int] = None

    # ---- preprocessing ----
    # preprocess_mode (if set) takes priority over preprocess_fn.
    preprocess_fn: typing.Callable[[str], str] = dataclasses.field(
        default=_stub_preprocess)
    preprocess_mode: typing.Optional[str] = None   # "none"|"normalize"|"replace"|"both"

    # ---- cache ----
    use_cache: bool = True
    cache_dir: str = "/content/cache"

    # ---- split ----
    eval_start_ratio: float = 0.8
    train_ratio: float = 0.3
    target_ratio: float = 0.3

    # ---- per-stage progressive data ----
    stage_train_ratios: typing.Optional[typing.Dict[int, float]] = None
    stage_min_ratios: typing.Optional[typing.Dict[int, float]] = None

    # ---- ablation ----
    use_oversampling: bool = True

    # ---- encoder ----
    encoder: str = "bert-base"                    # preset name OR hf path
    encoder_overrides: typing.Optional[typing.Dict] = None
    encoder_lora_r: int = 8
    encoder_lora_alpha: int = 16
    encoder_lora_targets: typing.Optional[typing.List[str]] = None
    encoder_quantize: bool = False
    encoder_last_layers: int = 1
    encoder_layer_agg: str = "last"     # "last"|"mean"|"weighted"|"attn_pool"
    encoder_pooling: str = "mean"       # "cls"|"mean"|"max"

    # ---- Llama 3 / 3.1 / 3.2 ----
    llama_path: str = "meta-llama/Meta-Llama-3-8B"
    llama_lora_r: int = 16
    llama_lora_alpha: int = 32
    decoder_lora_dropout: float = 0.1
    decoder_target_modules: list[str] = dataclasses.field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    decoder_head_pooling: Literal["last", "mean"] = "last"

    # ---- compute dtype ----
    compute_dtype: str = "bf16"         # "bf16"|"fp16"|"fp32"|"tf32"

    # ---- HF ----
    hf_token: typing.Optional[str] = None
    template_repo_id: typing.Optional[str] = None
    use_shared_template: bool = True

    # ---- collator ----
    max_content_len: int = 128

    # ---- time fusion ----
    use_time_features: bool = True
    fusion_type: str = "cross_attention"   # "cross_attention"|"concat"|"projector"|"none"
    use_time2vec: bool = True
    use_cyclic_time: bool = True
    time_emb_len: int = 16

    # ---- training ----
    grad_accum_steps: int = 4
    weight_decay: float = 0.05
    warmup_steps: int = 10
    grad_clip: float = 1.0

    # ---- scheduler ----
    scheduler_type: str = "polynomial"   # "polynomial"|"linear"|"cosine"|"constant"
    poly_power: float = 1.0
    min_lr: float = 1e-7
    cosine_min_ratio: float = 0.0

    # ---- stages ----
    n_epochs_stage1: int = 1
    n_epochs_stage2: int = 1
    n_epochs_stage3: int = 1
    n_epochs_stage4: int = 3
    lr_stage1: float = 5e-4
    lr_stage2: float = 5e-4
    lr_stage3: float = 5e-5
    lr_stage4: float = 2e-5
    template_max_samples: int = 1000

    # ---- eval ----
    eval_max_windows: typing.Optional[int] = 2000

    # ---- eval control ----
    # {1}    — только stage1 пропускается (baseline head на stage2 сохраняется)
    # {1,2}  — stage1+stage2 (быстрее, но теряете baseline head)
    # set()  — eval на всех стадиях (для абляций)
    skip_eval_stages: typing.Set[int] = dataclasses.field(
        default_factory=lambda: {1, 2})

    # ---- sampler ----
    sampler_max_oversample: float = 10.0
    sampler_min_minority: int = 50
    max_samples: typing.Optional[int] = None
    min_samples: typing.Optional[int] = None

    # ---- continue_learning ----
    resume_from: typing.Optional[str] = None
    resume_subdir: typing.Optional[str] = None
    resume_from_stage: str = "stage4"
    resume_next_stage: bool = False
    resume_n_epochs: typing.Optional[int] = None

    # ---- misc ----
    seed: int = 42
    save_dir: str = "/content/checkpoints"
    run_tag: typing.Optional[str] = None
    push_to_hub: bool = False
    hub_repo_id: typing.Optional[str] = None


# ============================================================
# Helpers
# ============================================================

_DATASET_CLS = {"bgl": BGL, "tbird": Tbird, "spirit": Spirit, "liberty": Liberty}
_STAGE_ORDER = ["stage1", "stage2", "stage3", "stage4"]
_STAGE_MODE = {"stage1": "llama", "stage2": "head",
               "stage3": "encoder", "stage4": "all"}


def _get_preprocess(cfg: Config):
    if cfg.preprocess_mode is not None:
        return get_preprocess_fn(cfg.preprocess_mode)
    return cfg.preprocess_fn


def _cfg_for_save(cfg: Config) -> dict:
    d = dataclasses.asdict(cfg)
    d.pop("preprocess_fn", None)
    return d


def _build_dataset(cfg: Config, preprocess=None):
    data_cls = _DATASET_CLS[cfg.dataset]
    max_lines = math.inf if cfg.max_lines is None else cfg.max_lines
    d = data_cls(
        window_size=cfg.win_size, step_size=cfg.step_size,
        max_lines=max_lines, start_line=cfg.start_line,
        end_line=cfg.end_line, use_in_colab=cfg.use_in_colab)
    cache_root = cfg.cache_dir if cfg.use_cache else None
    if cache_root:
        os.makedirs(cache_root, exist_ok=True)
    return SuperComputerDataset(
        d, cfg.anchor_step, train_ratio=cfg.eval_start_ratio,
        cache_dir=cache_root,
        preprocess_fn=preprocess or _get_preprocess(cfg))


class LogLLMCollator:
    def __init__(self, encoder_tokenizer, max_content_len: int = 128):
        self.tok = encoder_tokenizer
        self.max_content_len = max_content_len

    def __call__(self, batch):
        windows, times, labels = [], [], []
        for window, window_times, _raws, label in batch:
            if not window:
                continue
            windows.append(window)
            times.append(window_times)
            labels.append(float(label))
        if not windows:
            return None

        flat = [line for w in windows for line in w]
        positions, pos = [], 0
        for w in windows[:-1]:
            pos += len(w)
            positions.append(pos)

        enc_inputs = self.tok(
            flat, return_tensors="pt",
            max_length=self.max_content_len,
            padding=True, truncation=True)

        return {
            "bert_inputs": enc_inputs,
            "seq_positions": torch.tensor(positions, dtype=torch.long),
            "times": times,
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


class LimitedIterable(IterableDataset):
    def __init__(self, base, max_items):
        self.base, self.max_items = base, max_items

    def __iter__(self):
        if self.max_items is None:
            yield from self.base
            return
        for i, x in enumerate(self.base):
            if i >= self.max_items:
                break
            yield x

    def __len__(self):
        return len(self.base) if self.max_items is None \
               else min(len(self.base), self.max_items)


# ============================================================
# Scheduler
# ============================================================

def _build_scheduler(optimizer, total_steps: int, cfg: Config, max_lr: float):
    warmup = min(cfg.warmup_steps, max(1, total_steps // 10))
    sched = cfg.scheduler_type.lower()
    min_lr = float(cfg.min_lr)
    if cfg.cosine_min_ratio > 0:
        min_lr = float(max_lr) * float(cfg.cosine_min_ratio)

    if sched in ("polynomial", "linear"):
        power = 1.0 if sched == "linear" else float(cfg.poly_power)
        print(f"[sched] {sched} power={power} max_lr={max_lr:.2e} "
              f"min_lr={min_lr:.2e} warmup={warmup} total={total_steps}")
        return get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup,
            num_training_steps=total_steps,
            lr_end=min_lr, power=power)

    if sched == "cosine":
        def fn(step):
            if step < warmup:
                lr = min_lr + (max_lr - min_lr) * step / max(1, warmup)
            else:
                denom = max(1, total_steps - warmup)
                p = min(1.0, (step - warmup) / denom)
                lr = min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * p))
            return lr / max_lr
        print(f"[sched] cosine max_lr={max_lr:.2e} min_lr={min_lr:.2e} "
              f"warmup={warmup} total={total_steps}")
        return LambdaLR(optimizer, fn)

    if sched == "constant":
        print(f"[sched] constant max_lr={max_lr:.2e} warmup={warmup}")
        return get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup)

    raise ValueError(f"unknown scheduler_type={cfg.scheduler_type!r}")


# ============================================================
# HF helpers
# ============================================================

def _login_hf(cfg: Config) -> str:
    token = (cfg.hf_token
             or os.environ.get("HF_TOKEN")
             or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    from huggingface_hub import login, whoami, notebook_login
    if token:
        login(token=token, add_to_git_credential=False)
        print("[hf] logged in via token")
    else:
        try:
            whoami()
            print("[hf] already logged in")
        except Exception:
            notebook_login()
    return whoami()["name"]


def _resolve_template_repo(cfg: Config, user_name: str) -> str:
    return cfg.template_repo_id or f"{user_name}/llama3-8b-template"


def _template_exists(repo_id: str) -> bool:
    from huggingface_hub import HfApi
    try:
        files = HfApi().list_repo_files(repo_id, repo_type="model")
    except Exception:
        return False
    return any(f.startswith("llama_lora/") for f in files)


def _save_llama_template(model, repo_id: str):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    local = "/tmp/llama3_template"
    os.makedirs(os.path.join(local, "llama_lora"), exist_ok=True)
    model.llama.save_pretrained(os.path.join(local, "llama_lora"))
    api.upload_folder(folder_path=local, repo_id=repo_id,
                      path_in_repo=".", repo_type="model")
    print(f"[template] saved → {repo_id}")


def _load_llama_template(model, repo_id: str):
    from huggingface_hub import snapshot_download
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    local = snapshot_download(repo_id=repo_id, repo_type="model",
                               allow_patterns=["llama_lora/**"])
    fp = os.path.join(local, "llama_lora", "adapter_model.safetensors")
    if not os.path.exists(fp):
        raise FileNotFoundError(fp)
    state = load_file(fp)
    set_peft_model_state_dict(model.llama, state)
    print(f"[template] loaded ← {repo_id} ({len(state)} keys)")


# ============================================================
# Loaders
# ============================================================

def _build_loaders(cfg: Config, ds, val_ds, collator, stage_idx: int,
                    template_only: bool = False,
                    need_val: bool = True):
    ratio = cfg.train_ratio
    if cfg.stage_train_ratios and stage_idx in cfg.stage_train_ratios:
        ratio = cfg.stage_train_ratios[stage_idx]
    min_ratio = 0.0
    if cfg.stage_min_ratios and stage_idx in cfg.stage_min_ratios:
        min_ratio = cfg.stage_min_ratios[stage_idx]

    train_limit = int(ds.total_lines * ratio)
    safe_max = max(0, min(train_limit, val_ds.start_line - cfg.win_size))
    safe_min = min(int(ds.total_lines * min_ratio), safe_max) if min_ratio > 0 else 0

    if template_only:
        sampler = BalancedSampler(
            ds, target_ratio=0.5, seed=cfg.seed,
            max_samples=cfg.template_max_samples, min_samples=None,
            max_oversample_factor=cfg.sampler_max_oversample,
            min_minority_for_balance=10,
            valid_range=(safe_min, safe_max),
            use_oversampling=True, cache_dir=None)
    else:
        sampler = BalancedSampler(
            ds, target_ratio=cfg.target_ratio, seed=cfg.seed,
            max_samples=cfg.max_samples, min_samples=cfg.min_samples,
            max_oversample_factor=cfg.sampler_max_oversample,
            min_minority_for_balance=cfg.sampler_min_minority,
            valid_range=(safe_min, safe_max),
            use_oversampling=cfg.use_oversampling,
            cache_dir=cfg.cache_dir if cfg.use_cache else None)

    print(f"[loaders] stage{stage_idx} ratio={ratio} "
          f"safe=[{safe_min},{safe_max}) sampler_len={len(sampler)}"
          f"{' (template)' if template_only else ''}"
          f"{' (no val)' if not need_val else ''}")

    train_loader = DataLoader(
        ds, batch_size=cfg.batch_size, sampler=sampler,
        collate_fn=collator, num_workers=cfg.num_workers,
        drop_last=True, persistent_workers=cfg.num_workers > 0)

    if need_val:
        val_loader = DataLoader(
            LimitedIterable(val_ds, cfg.eval_max_windows),
            batch_size=cfg.batch_size, shuffle=False,
            collate_fn=collator, num_workers=cfg.num_workers)
    else:
        val_loader = None

    return train_loader, val_loader


# ============================================================
# Tests
# ============================================================

def test_preprocessing(cfg: Config, n_samples: int = 10):
    data_cls = _DATASET_CLS[cfg.dataset]
    max_lines = math.inf if cfg.max_lines is None else cfg.max_lines
    d = data_cls(
        window_size=cfg.win_size, step_size=cfg.step_size,
        max_lines=max_lines, start_line=cfg.start_line,
        end_line=cfg.end_line, use_in_colab=cfg.use_in_colab)
    preprocess = _get_preprocess(cfg)
    spec = resolve_encoder(cfg.encoder, cfg.encoder_overrides)
    tok = AutoTokenizer.from_pretrained(spec["path"])

    label = ("mode=" + repr(cfg.preprocess_mode)
             if cfg.preprocess_mode else "custom fn")
    print(f"\n{'='*90}\nPreprocessing test | {label} | "
          f"encoder={spec['path']}\nlog={d.path_to_log}\n{'='*90}\n")

    shown, skipped = 0, 0
    with open(d.path_to_log, "rb") as f:
        while shown < n_samples:
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("latin-1", errors="replace")
            dt, raw_cnt, lab = d.get_time_content_label(line)
            if not raw_cnt:
                skipped += 1
                continue
            proc = preprocess(raw_cnt)
            ids = tok(proc, truncation=False)["input_ids"]
            print(f"--- [{shown}] label={lab} time={dt} ---")
            print(f"RAW  : {raw_cnt!r}")
            print(f"PROC : {proc!r}")
            print(f"TOKENS ({len(ids)}): "
                  f"{tok.convert_ids_to_tokens(ids)[:30]}"
                  f"{'...' if len(ids) > 30 else ''}\n")
            shown += 1
    if skipped:
        print(f"(skipped {skipped} lines with empty Content)")


def test_dataloader(cfg: Config, n_windows: int = 3):
    ds = _build_dataset(cfg)
    val_ds = EvaluationDataset(ds)
    train_limit = int(ds.total_lines * cfg.train_ratio)
    safe_max = max(0, min(train_limit, val_ds.start_line - cfg.win_size))
    sampler = BalancedSampler(
        ds, target_ratio=cfg.target_ratio, seed=cfg.seed,
        max_samples=cfg.max_samples, min_samples=cfg.min_samples,
        max_oversample_factor=cfg.sampler_max_oversample,
        min_minority_for_balance=cfg.sampler_min_minority,
        valid_range=(0, safe_max),
        use_oversampling=cfg.use_oversampling, cache_dir=None)

    print(f"\n{'='*90}\nDataloader test | train_ratio={cfg.train_ratio} | "
          f"safe_max={safe_max} | sampler_total={len(sampler)}\n{'='*90}\n")

    for idx in list(iter(sampler))[:n_windows]:
        window, times, raws, label = ds[int(idx)]
        print(f"--- start={idx} label={label} len={len(window)} ---")
        for i, (w, t, _r) in enumerate(zip(window, times, raws)):
            if i >= 5:
                print(f"  ... ({len(window) - 5} more)")
                break
            print(f"  [{i:3d}] t={t}  {w!r}")
        print()


# ============================================================
# Eval
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, max_windows=None, desc="eval"):
    from sklearn.metrics import precision_recall_fscore_support
    model.eval()
    probs, ys, n_seen = [], [], 0
    pbar = tqdm.tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        if batch is None:
            continue
        logits = model(batch)
        probs.append(torch.sigmoid(logits.float()).cpu())
        ys.append(batch["labels"].cpu())
        n_seen += ys[-1].numel()
        pbar.set_postfix({"n": n_seen})
        if max_windows is not None and n_seen >= max_windows:
            break
    if not probs:
        return {"f1": 0., "precision": 0., "recall": 0., "n": 0}
    p = torch.cat(probs).numpy()
    y = torch.cat(ys).numpy().astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, (p >= 0.5).astype(int), average="binary", zero_division=0)
    return {"f1": float(f1), "precision": float(pr),
            "recall": float(rc), "n": len(y)}


# ============================================================
# Train one stage
# ============================================================

def train_one_stage(model, train_loader, val_loader, cfg, device,
                    n_epochs: int, lr: float, stage_name: str,
                    save_dir: str,
                    start_epoch: int = 0, start_step: int = 0,
                    optimizer_state=None, scheduler_state=None):
    if n_epochs <= 0:
        print(f"[{stage_name}] skip (n_epochs=0)")
        return start_epoch, start_step

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[{stage_name}] mode={_STAGE_MODE[stage_name]} "
          f"trainable={sum(p.numel() for p in trainable):,}")

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() <= 1:  # bias, LayerNorm weight/bias
            no_decay.append(p)
        elif "LayerNorm" in name or "layer_norm" in name.lower() or "layernorm" in name.lower() or "bias" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW([{'params':decay, 'weight_decay': cfg.weight_decay, 'lr':lr},  # для весов
        {'params': no_decay, 'weight_decay': 0.0, 'lr':lr}  # для смещений
    ])

    accum = max(1, cfg.grad_accum_steps)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))   # fix 1
    total_steps = steps_per_epoch * n_epochs
    scheduler = _build_scheduler(optimizer, total_steps, cfg, max_lr=lr)

    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print(f"[{stage_name}] optimizer state loaded")
        except Exception as e:
            print(f"[{stage_name}] optimizer state load failed: {e}")
    if scheduler_state is not None:
        try:
            scheduler.load_state_dict(scheduler_state)
            print(f"[{stage_name}] scheduler state loaded")
        except Exception as e:
            print(f"[{stage_name}] scheduler state load failed: {e}")

    criterion = nn.BCEWithLogitsLoss()
    global_step = start_step
    stage_num = _STAGE_ORDER.index(stage_name) + 1
    do_eval = (stage_num not in cfg.skip_eval_stages) and (val_loader is not None)

    for epoch_offset in range(n_epochs):
        epoch = start_epoch + epoch_offset
        model.train()
        running, n_b, acc_count = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        pbar = tqdm.tqdm(train_loader, desc=f"{stage_name} e{epoch}")
        for step_idx, batch in enumerate(pbar, 1):
            if batch is None:
                continue
            logits = model(batch)
            loss = criterion(logits, batch["labels"].to(device))
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            (loss / accum).backward()
            running += loss.item()
            n_b += 1
            acc_count += 1

            is_last = (step_idx == len(train_loader))
            if acc_count % accum == 0 or is_last:
                if cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                optimizer.step();
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                acc_count = 0

            pbar.set_postfix({"loss": f"{running / n_b:.4f}",
                              "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        # ---- eval (skippable per stage) ----
        if not do_eval:
            print(f"[{stage_name} e{epoch}] loss={running / max(n_b, 1):.4f} "
                  f"(eval skipped) elapsed={time.time() - t0:.0f}s")
        else:
            m = evaluate(model, val_loader, device,
                         max_windows=cfg.eval_max_windows,
                         desc=f"{stage_name} e{epoch} eval")
            print(f"[{stage_name} e{epoch}] loss={running / max(n_b, 1):.4f} "
                  f"F1={m['f1']:.4f} P={m['precision']:.4f} "
                  f"R={m['recall']:.4f} n={m['n']} "
                  f"elapsed={time.time() - t0:.0f}s")

    stage_dir = os.path.join(save_dir, stage_name)
    model.save(stage_dir)
    torch.save({
        "stage": stage_name,
        "epoch": start_epoch + n_epochs,
        "global_step": global_step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cfg": _cfg_for_save(cfg),
    }, os.path.join(stage_dir, "train_state.pt"))
    print(f"[save] {stage_dir}")

    if cfg.push_to_hub and cfg.hub_repo_id:
        from huggingface_hub import HfApi
        HfApi().upload_folder(
            folder_path=stage_dir, repo_id=cfg.hub_repo_id,
            path_in_repo=f"{cfg.run_tag}/{stage_name}",
            repo_type="model")
        print(f"[hf] uploaded {stage_name} → {cfg.hub_repo_id}")

    return start_epoch + n_epochs, global_step


# ============================================================
# run()
# ============================================================

def run(cfg: Config):
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")

    user_name = _login_hf(cfg)
    template_repo = _resolve_template_repo(cfg, user_name)
    print(f"[hf] user={user_name} template={template_repo}")

    if cfg.run_tag is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.run_tag = f"logllm_{cfg.dataset}_{cfg.encoder}_tr{cfg.train_ratio}_{ts}"
    save_dir = os.path.join(cfg.save_dir, cfg.run_tag)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[run] tag={cfg.run_tag} save_dir={save_dir}")

    ds = _build_dataset(cfg)
    val_ds = EvaluationDataset(ds)
    print(f"[data] total_lines={ds.total_lines} val_start={val_ds.start_line}")

    model = LogLLM(cfg, device)
    collator = LogLLMCollator(model.encoder_tokenizer, cfg.max_content_len)

    has_template = (cfg.use_shared_template
                    and _template_exists(template_repo))
    if has_template:
        print(f"[template] FOUND at {template_repo} → load, skip stage1")
        _load_llama_template(model, template_repo)
    else:
        print(f"[template] NOT FOUND → will train stage1 and push")

    # ---- stage1 (only if template missing) ----
    if not has_template:
        print(f"\n{'*' * 20} stage1 (llama) {'*' * 20}")
        model.set_train_mode("llama")
        need_val = 1 not in cfg.skip_eval_stages
        tr, va = _build_loaders(cfg, ds, val_ds, collator,
                                stage_idx=1, template_only=True,
                                need_val=need_val)
        train_one_stage(model, tr, va, cfg, device,
                        n_epochs=cfg.n_epochs_stage1, lr=cfg.lr_stage1,
                        stage_name="stage1", save_dir=save_dir)
        torch.cuda.empty_cache()
        try:
            _save_llama_template(model, template_repo)
        except Exception as e:
            print(f"[template] push failed: {e}")

    # ---- stages 2-4 ----
    for idx, name in zip([2, 3, 4], ["stage2", "stage3", "stage4"]):
        print(f"\n{'*' * 20} {name} ({_STAGE_MODE[name]}) {'*' * 20}")
        model.set_train_mode(_STAGE_MODE[name])
        need_val = idx not in cfg.skip_eval_stages
        tr, va = _build_loaders(cfg, ds, val_ds, collator, stage_idx=idx,
                                need_val=need_val)
        train_one_stage(model, tr, va, cfg, device,
                        n_epochs=getattr(cfg, f"n_epochs_{name}"),
                        lr=getattr(cfg, f"lr_{name}"),
                        stage_name=name, save_dir=save_dir)
        torch.cuda.empty_cache()

    print(f"\n[done] checkpoints → {save_dir}")
    return {"model": model, "dataset": ds, "val_dataset": val_ds,
            "cfg": cfg, "device": device, "save_dir": save_dir}


# ============================================================
# run_many() — несколько конфигов подряд (sequential)
# ============================================================

def run_many(configs: typing.Sequence[Config]):
    """Прогоняет run(cfg) для каждой конфигурации. Первый обучит template,
    остальные его загрузят."""
    results = {}
    for i, cfg in enumerate(configs):
        key = f"{cfg.dataset}__{cfg.encoder}"
        print(f"\n\n{'#' * 30} [{i+1}/{len(configs)}] {key} {'#' * 30}\n")
        results[key] = run(cfg)
    return results


# ============================================================
# continue_learning()
# ============================================================

def _download_from_hf(repo_id: str, subdir: str) -> str:
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo_id=repo_id, repo_type="model",
                               allow_patterns=[f"{subdir}/**"])
    return os.path.join(local, subdir)


def _resolve_resume_path(cfg: Config) -> str:
    if cfg.resume_from is None:
        raise ValueError("cfg.resume_from is None")
    if cfg.resume_subdir:
        return _download_from_hf(cfg.resume_from, cfg.resume_subdir)
    if not os.path.isdir(cfg.resume_from):
        raise FileNotFoundError(cfg.resume_from)
    return cfg.resume_from


def continue_learning(cfg: Config):
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")
    _login_hf(cfg)

    resume_path = _resolve_resume_path(cfg)
    resume_stage_dir = os.path.join(resume_path, cfg.resume_from_stage)
    state_path = os.path.join(resume_stage_dir, "train_state.pt")
    if not os.path.exists(state_path):
        raise FileNotFoundError(state_path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    print(f"[resume] {state_path} | epoch={state.get('epoch')} "
          f"step={state.get('global_step')}")

    if cfg.resume_next_stage:
        idx = _STAGE_ORDER.index(cfg.resume_from_stage)
        if idx + 1 >= len(_STAGE_ORDER):
            print(f"[resume] last stage — nothing to do")
            return None
        target = _STAGE_ORDER[idx + 1]
        start_epoch, start_step = 0, 0
        optimizer_state = scheduler_state = None
        print(f"[resume] {cfg.resume_from_stage} → {target} "
              f"(fresh optimizer)")
    else:
        target = cfg.resume_from_stage
        start_epoch = int(state.get("epoch", 0))
        start_step = int(state.get("global_step", 0))
        optimizer_state = state.get("optimizer")
        scheduler_state = state.get("scheduler")
        print(f"[resume] continue {target} from epoch={start_epoch} "
              f"step={start_step}")

    n_epochs = (cfg.resume_n_epochs
                if cfg.resume_n_epochs is not None
                else getattr(cfg, f"n_epochs_{target}"))
    lr = getattr(cfg, f"lr_{target}")
    stage_idx = _STAGE_ORDER.index(target) + 1

    ds = _build_dataset(cfg)
    val_ds = EvaluationDataset(ds)

    model = LogLLM(cfg, device)
    model.load(resume_stage_dir)
    collator = LogLLMCollator(model.encoder_tokenizer, cfg.max_content_len)

    model.set_train_mode(_STAGE_MODE[target])
    need_val = stage_idx not in cfg.skip_eval_stages
    tr, va = _build_loaders(cfg, ds, val_ds, collator, stage_idx=stage_idx,
                            template_only=(target == "stage1"),
                            need_val=need_val)

    if cfg.run_tag is None:
        cfg.run_tag = f"continue_{target}_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    save_dir = os.path.join(cfg.save_dir, cfg.run_tag)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'*' * 20} {target} ({_STAGE_MODE[target]}) {'*' * 20}")
    end_epoch, end_step = train_one_stage(
        model, tr, va, cfg, device,
        n_epochs=n_epochs, lr=lr, stage_name=target, save_dir=save_dir,
        start_epoch=start_epoch, start_step=start_step,
        optimizer_state=optimizer_state, scheduler_state=scheduler_state)
    print(f"\n[done] epoch={end_epoch} step={end_step} "
          f"checkpoints → {save_dir}")
    return {"model": model, "dataset": ds, "val_dataset": val_ds,
            "cfg": cfg, "device": device, "save_dir": save_dir}


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train",
                        choices=["train", "continue", "test_preproc",
                                 "test_loader"])
    parser.add_argument("--dataset", default="liberty",
                        choices=list(_DATASET_CLS.keys()))
    parser.add_argument("--encoder", default="bert-base",
                        help=f"preset name or hf path. presets: {list(ENCODER_PRESETS)}")
    parser.add_argument("--llama", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--preprocess", default=None,
                        choices=["none", "normalize", "replace", "both"])
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--template-repo", default=None)
    parser.add_argument("--compute-dtype", default="bf16",
                        choices=["bf16", "fp16", "fp32", "tf32"])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--resume-subdir", default=None)
    parser.add_argument("--resume-stage", default="stage4",
                        choices=_STAGE_ORDER)
    parser.add_argument("--resume-next", action="store_true")
    parser.add_argument("--resume-epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = Config(
        dataset=args.dataset,
        encoder=args.encoder,
        llama_path=args.llama,
        preprocess_mode=args.preprocess,
        hf_token=args.hf_token,
        template_repo_id=args.template_repo,
        compute_dtype=args.compute_dtype,
        resume_from=args.resume_from,
        resume_subdir=args.resume_subdir,
        resume_from_stage=args.resume_stage,
        resume_next_stage=args.resume_next,
        resume_n_epochs=args.resume_epochs,
    )

    if args.mode == "test_preproc":
        test_preprocessing(cfg, n_samples=args.n)
    elif args.mode == "test_loader":
        test_dataloader(cfg, n_windows=args.n)
    elif args.mode == "continue":
        continue_learning(cfg)
    else:
        run(cfg)
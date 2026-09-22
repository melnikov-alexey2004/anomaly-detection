import math
import os.path
import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForCausalLM,
    BitsAndBytesConfig, DynamicCache,
)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

def stack_and_pad_left(tensors):
    max_len = max(t.shape[0] for t in tensors)
    padded, masks = [], []
    for t in tensors:
        pad = max_len - t.shape[0]
        padded.append(F.pad(t, (0, 0, pad, 0)))
        masks.append(torch.cat([
            torch.zeros(pad, dtype=torch.long),
            torch.ones(t.shape[0], dtype=torch.long),
        ]))
    return torch.stack(padded), torch.stack(masks)

class Time2Vec(nn.Module):
    # t2v(τ)[0] = w0·τ + b0
    # t2v(τ)[i] = sin(w_i·τ + b_i)
    # https://arxiv.org/html/1907.05321

    def __init__(self, k: int = 16):
        super().__init__()
        self.k = k
        self.w0 = nn.Parameter(torch.randn(1) * 0.01)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(k) * 0.01)
        self.b = nn.Parameter(torch.zeros(k))

    def forward(self, t): # [N,]
        lin = (self.w0 * t + self.b0).unsqueeze(-1) # [N, 1]
        per = torch.sin(t.unsqueeze(-1) * self.w + self.b) # [N, 1] @ [k] + [k]
        return torch.cat([lin, per], dim=-1) # [N, k+1]


def cyclic_time_features(times):
    # https://arxiv.org/pdf/2411.15250
    feats = []
    for t in times:
        h, m = t.hour / 24.0, t.minute / 60.0
        # jan 1- feb 2 - ... - dec 12 - jan 1 (13)
        #     0      1             11    12
        dow, mon = t.weekday() / 7.0, (t.month - 1) / 12.0
        dom = (t.day - 1) / 31.0
        feats.append([
            math.sin(2*math.pi*h),   math.cos(2*math.pi*h),
            math.sin(2*math.pi*m),   math.cos(2*math.pi*m),
            math.sin(2*math.pi*dow), math.cos(2*math.pi*dow),
            math.sin(2*math.pi*mon), math.cos(2*math.pi*mon),
            math.sin(2*math.pi*dom), math.cos(2*math.pi*dom),
        ])
    return torch.tensor(feats, dtype=torch.float32)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

class Projector(nn.Module):
    def __init__(self, in_dim, out_dim, device):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, device=device)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)
        self.act = nn.LeakyReLU(negative_slope=0.01)
        # слишком большая дисперсия была  бы для пространства эмбеддингов
        self.norm = nn.LayerNorm(out_dim, device=device)
        # только Linear в bf16
        # layer norm остаётся в fp32
        self.linear = self.linear.to(torch.bfloat16)

    def forward(self, x):
        x = self.linear(x.to(torch.bfloat16))
        x = self.act(x)
        x = self.norm(x.float())
        return x.to(torch.bfloat16)

class Encoder(nn.Module):
    def __init__(
        self,
        encoder_path,
        decoder_path,
        ft_path=None,
        is_train_mode=True,
        device=torch.device("cuda:0"),
        max_content_len=4096, # в один чанк
        time2vec_k=16,
        n_layers_agg=4,
    ):
        super().__init__()
        self.max_content_len = max_content_len
        self.time2vec_k = time2vec_k
        self.n_layers_agg = n_layers_agg
        self.device = device

        self.decoder_tokenizer = AutoTokenizer.from_pretrained(
            decoder_path, padding_side="right", trust_remote_code=True
        )
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token
        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder_path, quantization_config=bnb_config,
            low_cpu_mem_usage=True, device_map=device, trust_remote_code=True,
        )
        self.decoder_hidden = self.decoder.config.hidden_size

        self.encoder_tokenizer = AutoTokenizer.from_pretrained(
            encoder_path, trust_remote_code=True
        )
        self.encoder = AutoModel.from_pretrained(
            encoder_path, quantization_config=bnb_config,
            low_cpu_mem_usage=True, device_map=device, trust_remote_code=True,
        )
        self.encoder_hidden = self.encoder.config.hidden_size
        self.sep_id = self.encoder_tokenizer.sep_token_id
        self.cls_id = self.encoder_tokenizer.cls_token_id

        self.time2vec = Time2Vec(time2vec_k).to(device)
        self.time_feat_dim = (time2vec_k + 1) + 10

        self.projector = Projector(self.encoder_hidden + self.time_feat_dim,
                                   self.decoder_hidden,
                                   device)

        self.instruc_tokens = self.decoder_tokenizer(
            ["Below is a sequence of system log messages:",
             ". Is this sequence normal or anomalous? \n"],
            return_tensors="pt", padding=True,
        ).to(device)

        if ft_path is not None:
            self._load_ft(ft_path, is_train_mode)
        else:
            self._init_peft()

    def _init_peft(self):
        # peft оборачивает модель. которая доступна по аттр .model
        self.encoder = get_peft_model(self.encoder, LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=4, lora_alpha=32, lora_dropout=0.01,
            target_modules=["query", "value"],
        ))
        self.decoder = get_peft_model(self.decoder, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none", task_type=TaskType.CAUSAL_LM,
        ))

    def _load_ft(self, ft_path, is_trainable):
        self.encoder = PeftModel.from_pretrained(
            self.encoder, os.path.join(ft_path, "encoder_ft"), is_trainable=is_trainable)
        self.decoder = PeftModel.from_pretrained(
            self.decoder, os.path.join(ft_path, "decoder_ft"), is_trainable=is_trainable)
        self.projector.load_state_dict(torch.load(
            os.path.join(ft_path, "projector.pt"), map_location=self.device, weights_only=True))
        self.time2vec.load_state_dict(torch.load(
            os.path.join(ft_path, "time2vec.pt"), map_location=self.device, weights_only=True))

    def save_ft_model(self, path):
        os.makedirs(path, exist_ok=True)
        self.encoder.save_pretrained(os.path.join(path, "encoder_ft"))
        self.decoder.save_pretrained(os.path.join(path, "decoder_ft"))
        torch.save(self.projector.state_dict(), os.path.join(path, "projector.pt"))
        torch.save(self.time2vec.state_dict(), os.path.join(path, "time2vec.pt"))

    def set_train_only_projector(self):
        for p in self.projector.parameters(): p.requires_grad = True
        for p in self.time2vec.parameters(): p.requires_grad = True
        for p in self.encoder.parameters():  p.requires_grad = False
        for p in self.decoder.parameters():  p.requires_grad = False

    def set_train_projector_and_encoder(self):
        for p in self.projector.parameters(): p.requires_grad = True
        for p in self.time2vec.parameters(): p.requires_grad = True
        for n, p in self.encoder.named_parameters():
            p.requires_grad = "lora" in n
        for p in self.decoder.parameters(): p.requires_grad = False

    def set_finetuning_all(self):
        for p in self.projector.parameters(): p.requires_grad = True
        for p in self.time2vec.parameters(): p.requires_grad = True
        for n, p in self.encoder.named_parameters():
            p.requires_grad = "lora" in n
        for n, p in self.decoder.named_parameters():
            p.requires_grad = "lora" in n

    def _build_chunks(self, batch_log_ids):
        # batch_log_ids: List[List[List[int]]]
        # [окно][лог][token_id]

        # all_chunks: List[(ids, offsets)]
        # где offsets = [(log_idx, start, end)]
        # window_ranges: List[(chunk_start, chunk_end)]
        # window_log_counts: List[int]
        all_chunks, window_ranges, window_log_counts = [], [], []
        sep, cls, L = self.sep_id, self.cls_id, self.max_content_len

        for window in batch_log_ids:
            c_start = len(all_chunks)
            cur_ids, cur_offsets, cur_len = [cls], [], 1

            for log_idx, ids in enumerate(window):
                max_log = L - 2
                if len(ids) > max_log:
                    ids = ids[:max_log]
                if not ids:
                    continue

                needed = len(ids) + 1
                if cur_len + needed > L:
                    if cur_offsets:
                        all_chunks.append((cur_ids, cur_offsets))
                    cur_ids, cur_offsets, cur_len = [cls], [], 1

                start = cur_len
                cur_ids.extend(ids)
                cur_len += len(ids)
                end = cur_len
                cur_offsets.append((log_idx, start, end))
                cur_ids.append(sep)
                cur_len += 1

            if cur_offsets:
                all_chunks.append((cur_ids, cur_offsets))

            window_ranges.append((c_start, len(all_chunks)))
            window_log_counts.append(len(window))

        return all_chunks, window_ranges, window_log_counts

    def _encode_and_pool(self, all_chunks, window_ranges, window_log_counts):
        max_chunk = max(len(c[0]) for c in all_chunks)
        input_ids  = torch.zeros(len(all_chunks), max_chunk, dtype=torch.long)
        attn_mask  = torch.zeros_like(input_ids)
        global_msk = torch.zeros_like(input_ids)

        for i, (ids, _) in enumerate(all_chunks):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            attn_mask[i, :n] = 1
            global_msk[i, 0] = 1

        input_ids  = input_ids.to(self.device)
        attn_mask  = attn_mask.to(self.device)
        global_msk = global_msk.to(self.device)

        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attn_mask,
            global_attention_mask=global_msk,
            output_hidden_states=True,
        )
        hidden = torch.stack(out.hidden_states[-self.n_layers_agg:], dim=0).mean(0)

        log_embs = []
        for w_idx, (c_start, c_end) in enumerate(window_ranges):
            per_log = [None] * window_log_counts[w_idx]
            for c in range(c_start, c_end):
                _, offsets = all_chunks[c]
                for log_idx, s, e in offsets:
                    per_log[log_idx] = hidden[c, s:e].mean(dim=0)
            # падинг для логов которые
            for j, e in enumerate(per_log):
                if e is None:
                    per_log[j] = torch.zeros(self.encoder_hidden, device=self.device)
            log_embs.extend(per_log)

        return torch.stack(log_embs, dim=0) # [total_logs, H_enc]

    def _time_features(self, times_flat, window_log_counts):
        cyclic = cyclic_time_features(times_flat).to(self.device)

        t2v_list, pos = [], 0
        for n in window_log_counts:
            if n == 0:
                continue
            t0 = times_flat[pos]
            deltas = torch.tensor(
                [(times_flat[pos + i] - t0).total_seconds() / 3600.0 for i in range(n)],
                device=self.device, dtype=torch.float32,
            )
            t2v_list.append(self.time2vec(deltas))
            pos += n

        t2v = torch.cat(t2v_list, dim=0)
        return torch.cat([t2v, cyclic], dim=-1).to(torch.bfloat16)

    def _decoder_embed(self, ids):
        base = self.decoder
        if isinstance(base, PeftModel):
            base = base.base_model.model
        if hasattr(base, "model") and hasattr(base.model, "embed_tokens"):
            return base.model.embed_tokens(ids)
        return base.get_input_embeddings()(ids)

    def _ins_emb(self):
        if not hasattr(self, "_ins_cache"):
            emb = self._decoder_embed(self.instruc_tokens["input_ids"])
            mask = self.instruc_tokens["attention_mask"].bool()
            prefix_ids = self.decoder_tokenizer(
                "The sequence is", return_tensors="pt")["input_ids"][0, 1:].to(self.device)
            self._ins_cache = {
                "ins1": emb[0][mask[0]],
                "ins2": emb[1][mask[1]][1:],
                "prefix": self._decoder_embed(prefix_ids),
            }
        return self._ins_cache

    def _log_embeddings(self, batch_log_ids, times_flat):
        all_chunks, wranges, wcounts = self._build_chunks(batch_log_ids)
        enc_embs = self._encode_and_pool(all_chunks, wranges, wcounts)
        time_embs = self._time_features(times_flat, wcounts)
        combined = torch.cat([enc_embs.to(torch.bfloat16), time_embs], dim=-1)
        return self.projector(combined), wcounts

    def train_helper(self, batch_log_ids, times_flat, labels):
        B = len(labels)
        log_proj, wcounts = self._log_embeddings(batch_log_ids, times_flat)

        prefix = "The sequence is "
        answer_tokens = self.decoder_tokenizer(
            [prefix + s + "." for s in labels], padding=True, return_tensors="pt"
        ).to(self.device)

        target_ids = torch.cat([
            answer_tokens["input_ids"][:, 1:],
            torch.full((B, 1), self.decoder_tokenizer.eos_token_id, device=self.device),
        ], dim=-1)
        target_atts = answer_tokens["attention_mask"].bool()

        ans_ids = answer_tokens["input_ids"][:, 1:]
        ans_atts = answer_tokens["attention_mask"].bool()[:, 1:]
        ans_embs = self._decoder_embed(ans_ids)

        ins = self._ins_emb()
        prompts, target_lens, pos = [], [], 0
        for b, n in enumerate(wcounts):
            logs_cat = log_proj[pos:pos + n]; pos += n
            ans_cat = ans_embs[b][ans_atts[b]]
            full = torch.cat([ins["ins1"], logs_cat, ins["ins2"], ans_cat], dim=0)
            prompts.append(full)
            target_lens.append(ans_cat.shape[0])

        inputs_embeds, attn = stack_and_pad_left(prompts)
        attn = attn.to(self.device)
        label_mask = attn.clone()
        for i in range(label_mask.shape[0]):
            label_mask[i, : -target_lens[i] - 1] = 0
        label_mask = label_mask.bool()

        out = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attn).logits
        return out[label_mask], target_ids[target_atts]

    @torch.no_grad()
    def forward(self, batch_log_ids, times_flat):
        log_proj, wcounts = self._log_embeddings(batch_log_ids, times_flat)
        ins = self._ins_emb()

        prompts, pos = [], 0
        for n in wcounts:
            logs_cat = log_proj[pos:pos + n]; pos += n
            prompts.append(torch.cat([ins["ins1"], logs_cat, ins["ins2"], ins["prefix"]], dim=0))

        inputs_embeds, attn = stack_and_pad_left(prompts)
        attn = attn.to(self.device)

        B = len(wcounts)
        pad_id = self.decoder_tokenizer.pad_token_id
        eos_id = self.decoder_tokenizer.eos_token_id
        eos_t = torch.tensor([eos_id], device=self.device)

        unfinished = torch.ones(B, dtype=torch.long, device=self.device)
        cache = DynamicCache()
        answer = []

        while True:
            if len(cache) == 0:
                out = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attn,
                                   past_key_values=cache, use_cache=True)
            else:
                out = self.decoder(inputs_embeds=next_emb[:, None, :], attention_mask=attn,
                                   past_key_values=cache, use_cache=True)
            nxt = torch.argmax(out.logits[:, -1, :], dim=-1)
            nxt = nxt * unfinished + pad_id * (1 - unfinished)
            answer.append(nxt)
            next_emb = self._decoder_embed(nxt)
            attn = torch.cat([attn, unfinished[:, None]], dim=1)
            unfinished = unfinished.mul(
                nxt.tile(eos_t.shape[0], 1).ne(eos_t.unsqueeze(1)).prod(dim=0))
            if unfinished.max() == 0 or len(answer) > 5:
                break

        return torch.stack(answer, dim=1)
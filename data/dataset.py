"""Phase 1.3 + 1.5: VQA dataset, sequence layout, loss mask, and collate.

Sequence: [BOS] <question tokens> <answer tokens> [EOS]
The image does NOT enter the token sequence (it enters via cross-attention).
Loss mask: label = -100 for BOS + question; supervise only answer span + EOS.
"""
import json
import os

import torch
from torch.utils.data import Dataset


def _key(image_id):
    return os.path.splitext(str(image_id))[0].lower()


class VQADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, cfg, feature_dir):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))
        self.tok = tokenizer
        self.feature_dir = feature_dir
        self.max_q = cfg["tokenizer"]["max_question_tokens"]
        self.max_a = cfg["tokenizer"]["max_answer_tokens"]
        self.eos_id = cfg["model"]["eos_id"]
        self.bos_id = (tokenizer.bos_token_id
                       if tokenizer.bos_token_id is not None else 1)

    def __len__(self):
        return len(self.rows)

    def _encode(self, text, max_len):
        ids = self.tok.encode(text, add_special_tokens=False)
        return ids[:max_len]

    def __getitem__(self, i):
        r = self.rows[i]
        q_ids = self._encode(str(r["question"]), self.max_q)
        a_ids = self._encode(str(r["answer"]), self.max_a)

        input_ids = [self.bos_id] + q_ids + a_ids + [self.eos_id]
        # supervise only answer span + EOS
        labels = ([-100] * (1 + len(q_ids))) + a_ids + [self.eos_id]
        assert len(input_ids) == len(labels)

        feat_path = os.path.join(self.feature_dir, _key(r["image_id"]) + ".pt")
        feats = torch.load(feat_path, map_location="cpu")  # [196,768] fp16

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_features": feats,
            # kept for eval / generation and per-category metrics
            "question": str(r["question"]),
            "answer": str(r["answer"]),
            "question_type": r["question_type"],
            "complexity": r["complexity"],
            "prompt_len": 1 + len(q_ids),  # BOS + question (where the answer starts)
        }


def make_collate(pad_id):
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        B = len(batch)
        input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
        labels = torch.full((B, maxlen), -100, dtype=torch.long)
        attn = torch.zeros((B, maxlen), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            input_ids[i, :L] = b["input_ids"]
            labels[i, :L] = b["labels"]
            attn[i, :L] = 1
        image_features = torch.stack([b["image_features"] for b in batch])  # [B,196,768]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn,
            "image_features": image_features,
            "questions": [b["question"] for b in batch],
            "answers": [b["answer"] for b in batch],
            "question_types": [b["question_type"] for b in batch],
            "complexities": [b["complexity"] for b in batch],
            "prompt_lens": [b["prompt_len"] for b in batch],
        }
    return collate

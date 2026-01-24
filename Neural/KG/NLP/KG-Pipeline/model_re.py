from __future__ import annotations

from typing import Any

import torch

def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
        return torch.device("mps")
    return torch.device("cpu")

def id_to_label(model: Any, pred_id: int) -> str:
    id2label = model.config.id2label
    if isinstance(id2label, dict):
        if pred_id in id2label:
            return str(id2label[pred_id])
        s = str(pred_id)
        if s in id2label:
            return str(id2label[s])
        # last resort: scan keys that might be strings of ints
        for k, v in id2label.items():
            try:
                if int(k) == pred_id:
                    return str(v)
            except Exception:
                pass
        raise KeyError(f"pred_id={pred_id} not found in id2label keys (sample)={list(id2label)[:10]}")
    # list/tuple
    return str(id2label[pred_id])

@torch.no_grad()
def predict_one(
    marked_text: str,
    *,
    model,
    tokenizer,
    device: torch.device,
    top_k: int = 5,
    max_length: int = 256,
    temperature: float = 1.0,
):
    enc = tokenizer(marked_text, truncation=True, max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    logits = out.logits  # [1, C]
    if temperature and temperature != 1.0:
        logits = logits / float(temperature)
    probs = torch.softmax(logits, dim=-1).squeeze(0)  # [C]

    conf = float(probs.max().item())
    pred_id = int(torch.argmax(probs).item())

    k = min(int(top_k), int(probs.shape[-1]))
    top_probs, top_ids = torch.topk(probs, k=k)
    topk_ids = [(int(i.item()), float(p.item())) for i, p in zip(top_ids, top_probs)]
    return pred_id, conf, topk_ids

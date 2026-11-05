import json
from functools import lru_cache
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from normalizer import normalize
except ImportError:
    normalize = None

MODELS_DIR = Path(__file__).resolve().parent / "ml" / "webapp_models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=None)
def _load(task):
    """Prothom request-e ekbar load hobe, tarpor memory-te thakbe."""
    path = MODELS_DIR / task
    with open(path / "labels.json", encoding="utf-8") as f:
        meta = json.load(f)

    tok = AutoTokenizer.from_pretrained(meta["base_model"])
    base = AutoModelForSequenceClassification.from_pretrained(
        meta["base_model"], num_labels=len(meta["labels"])
    )
    model = PeftModel.from_pretrained(base, str(path))
    model.to(DEVICE).eval()
    return meta, tok, model


def predict(task, text):
    meta, tok, model = _load(task)
    if normalize:
        text = normalize(text)

    max_len = meta["max_len"]
    truncated = len(tok(text)["input_ids"]) > max_len

    enc = tok(text, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].cpu().tolist()

    best = max(range(len(probs)), key=probs.__getitem__)
    return {
        "label": meta["labels"][best],
        "confidence": probs[best],
        "probabilities": dict(zip(meta["labels"], probs)),
        "truncated": truncated,
        "model_info": {
            "test_macro_f1": meta.get("test_macro_f1"),
            "train_samples": meta.get("train_samples"),
        },
    }
"""Loads BanglaBERT + one LoRA adapter per task and predicts. Models are loaded lazily on first use."""
import json
import threading

from django.conf import settings

BASE_MODEL = 'csebuetnlp/banglabert'

TASK_INFO = {
    'sentiment': {
        'title': 'Sentiment',
        'hint': 'Book-review sentiment: negative, neutral or positive.',
        'example': 'বইটি খুব সুন্দর, পড়ে অনেক ভালো লেগেছে।',
        'warning': 'Trained on book reviews (BanglaBook). The neutral and negative classes are rare, so they are predicted less reliably.',
    },
    'hate': {
        'title': 'Hate speech',
        'hint': 'Social-media comment: hate speech or not.',
        'example': 'আজকের আবহাওয়া খুব সুন্দর।',
        'warning': 'Trained on BD-SHS comments. It can be wrong, so it must not be used for real moderation decisions.',
    },
    'fakenews': {
        'title': 'Fake news',
        'hint': 'News text: authentic or fake.',
        'example': 'প্রধানমন্ত্রী আজ নতুন সেতুর উদ্বোধন করেছেন।',
        'warning': 'Trained on 2018-2019 news (BanFakeNews) and reads only the first 128 tokens. It does not check facts, so do not use it to judge real news.',
    },
}

_cache = {}
_lock = threading.Lock()


def _load(task):
    # heavy imports happen here so that the server starts fast and the views can be tested without torch
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel

    folder = settings.MODELS_DIR / task
    meta_path = folder / 'labels.json'
    if not meta_path.exists():
        raise FileNotFoundError(f'model files for "{task}" not found in {folder}')
    meta = json.loads(meta_path.read_text(encoding='utf-8'))

    tokenizer = AutoTokenizer.from_pretrained(meta.get('base_model', BASE_MODEL))
    base = AutoModelForSequenceClassification.from_pretrained(meta.get('base_model', BASE_MODEL), num_labels=len(meta['labels']))
    model = PeftModel.from_pretrained(base, str(folder))
    model.eval()
    try:
        from normalizer import normalize      # csebuetnlp text normalizer, used during training
    except Exception:
        normalize = None
    return {'tokenizer': tokenizer, 'model': model, 'meta': meta, 'normalize': normalize, 'torch': torch}


def get_bundle(task):
    with _lock:
        if task not in _cache:
            _cache[task] = _load(task)
        return _cache[task]


def predict(task, text):
    b = get_bundle(task)
    torch, tok, meta = b['torch'], b['tokenizer'], b['meta']
    clean = b['normalize'](text) if b['normalize'] else text
    max_len = int(meta.get('max_len', 128))
    n_tokens = len(tok(clean, add_special_tokens=True)['input_ids'])
    enc = tok(clean, truncation=True, max_length=max_len, return_tensors='pt')
    with torch.inference_mode():
        probs = torch.softmax(b['model'](**enc).logits, dim=-1)[0].tolist()
    labels = meta['labels']
    best = max(range(len(labels)), key=lambda i: probs[i])
    return {
        'task': task,
        'label': labels[best],
        'confidence': round(probs[best], 4),
        'probabilities': {labels[i]: round(probs[i], 4) for i in range(len(labels))},
        'truncated': n_tokens > max_len,
        'model_info': {k: meta[k] for k in ('test_macro_f1', 'test_accuracy', 'train_samples') if k in meta},
    }

import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .model_loader import predict

MAX_CHARS = 2000

TASKS = {
    "sentiment": {
        "title": "Sentiment",
        "hint": "Predicts whether the text is negative, neutral or positive.",
        "example": "বইটি অসাধারণ, পড়ে খুব ভালো লেগেছে।",
        "warning": "Sentiment predictions can be wrong, especially for sarcasm.",
    },
    "hate": {
        "title": "Hate speech",
        "hint": "Checks whether the text contains hate speech.",
        "example": "আপনার মতামতের সাথে আমি একমত নই, তবে আলোচনা চালিয়ে যাওয়া যাক।",
        "warning": "This should not be the only basis for content moderation.",
    },
    "fakenews": {
        "title": "Fake news",
        "hint": "Estimates whether a news text is fake or authentic.",
        "example": "ঢাকায় আগামীকাল থেকে নতুন মেট্রোরেল সূচি কার্যকর হবে বলে জানিয়েছে কর্তৃপক্ষ।",
        "warning": "This is not fact-checking; it only looks at writing patterns.",
    },
}


def home(request):
    return render(request, "classifier/index.html", {"tasks": TASKS, "max_chars": MAX_CHARS})


@require_POST
def predict_view(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)

    task = data.get("task")
    text = (data.get("text") or "").strip()

    if task not in TASKS:
        return JsonResponse({"error": "Unknown task."}, status=400)
    if not text:
        return JsonResponse({"error": "Please enter some text."}, status=400)
    if len(text) > MAX_CHARS:
        return JsonResponse({"error": f"Text is too long (max {MAX_CHARS} characters)."}, status=400)

    return JsonResponse(predict(task, text))
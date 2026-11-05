# Bangla Text Classifier (Django demo)

BanglaBERT + one small LoRA adapter per task (sentiment, hate speech, fake news).
The base model (about 440 MB) is downloaded from Hugging Face on the first prediction. Each adapter is only a few MB.

## Run it

1. Python 3.10 or newer. Create a virtual environment and install the packages:

       python -m venv venv
       venv\Scripts\activate          (Windows)   or   source venv/bin/activate   (Mac/Linux)
       pip install -r requirements.txt

   `torch` can be large. If the install is slow, install the CPU build from https://pytorch.org first.

2. Get the trained adapters. In the Colab notebook run Cell 12, then download `webapp_models.zip`
   from Google Drive (MyDrive/bangla_peft/). Unzip it into this folder so you have:

       webapp_models/hate/labels.json
       webapp_models/sentiment/labels.json
       webapp_models/fakenews/labels.json

3. Start the server and open http://127.0.0.1:8000

       python manage.py runserver

The first request for each task is slow (model loading). Later requests take well under a second on a normal CPU.

## Tests (no model files or GPU needed)

    python manage.py test

## API

    POST /api/predict   {"task": "hate", "text": "..."}
    task is one of: sentiment, hate, fakenews

## Notes

- This is a research demo. The hate-speech and fake-news models make mistakes and must not be used for real decisions.
- To change where the adapters live, set the environment variable BANGLA_MODELS_DIR.
- For a public deployment set DJANGO_DEBUG=0, a real DJANGO_SECRET_KEY and DJANGO_ALLOWED_HOSTS, and run behind gunicorn.

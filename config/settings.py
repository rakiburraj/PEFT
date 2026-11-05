import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'dev-only-change-me')
DEBUG = os.environ.get('DJANGO_DEBUG', '1') == '1'
ALLOWED_HOSTS = os.environ.get('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "classifier",
]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
]
ROOT_URLCONF = 'config.urls'
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]
WSGI_APPLICATION = 'config.wsgi.application'

DATABASES = {}          # the app stores nothing, so no database is needed
STATIC_URL = 'static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# folder that contains hate/, sentiment/ and fakenews/ (unzip webapp_models.zip here)
MODELS_DIR = Path(os.environ.get('BANGLA_MODELS_DIR', BASE_DIR / 'webapp_models'))
MAX_TEXT_CHARS = 3000

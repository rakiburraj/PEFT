import json
from unittest import mock

from django.test import SimpleTestCase


class PredictApiTests(SimpleTestCase):
    url = '/api/predict'

    def post(self, data):
        return self.client.post(self.url, data=json.dumps(data), content_type='application/json')

    def test_index_page_loads(self):
        r = self.client.get('/')
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, 'Bangla Text Classifier')

    def test_rejects_get(self):
        self.assertEqual(self.client.get(self.url).status_code, 405)

    def test_bad_json(self):
        r = self.client.post(self.url, data='not json', content_type='application/json')
        self.assertEqual(r.status_code, 400)

    def test_unknown_task_and_empty_text(self):
        self.assertEqual(self.post({'task': 'nope', 'text': 'x'}).status_code, 400)
        self.assertEqual(self.post({'task': 'hate', 'text': '   '}).status_code, 400)

    def test_text_too_long(self):
        self.assertEqual(self.post({'task': 'hate', 'text': 'অ' * 5000}).status_code, 400)

    def test_success_uses_model(self):
        fake = {'task': 'hate', 'label': 'not hate', 'confidence': 0.9, 'probabilities': {'not hate': 0.9, 'hate': 0.1},
                'truncated': False, 'model_info': {}}
        with mock.patch('classifier.inference.predict', return_value=fake):
            r = self.post({'task': 'hate', 'text': 'আজকের আবহাওয়া সুন্দর'})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['label'], 'not hate')

    def test_missing_model_files_give_503(self):
        with mock.patch('classifier.inference.predict', side_effect=FileNotFoundError('missing')):
            self.assertEqual(self.post({'task': 'hate', 'text': 'টেক্সট'}).status_code, 503)

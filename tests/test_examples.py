import json
import os
import tempfile
import unittest
from pathlib import Path
os.environ['EMBEDDING_BACKEND']='lexical'
from fastapi.testclient import TestClient
from app.config import Config
from app.main import create_app

class ExampleEvaluationTests(unittest.TestCase):
    def test_all_40_synthetic_cases_in_lexical_mode(self):
        examples=Path(__file__).resolve().parents[2]/'examples'
        with tempfile.TemporaryDirectory() as tmp:
            config=Config(data_dir=Path(tmp),embedding_backend='lexical',answer_mode='extractive',app_token='')
            with TestClient(create_app(config)) as c:
                upload=c.post('/api/documents',files={'file':('company_policy.csv',(examples/'company_policy.csv').read_bytes())})
                self.assertEqual(upload.status_code,201)
                c.put('/api/settings',json={'top_k':5,'min_similarity':0.1})
                result=c.post('/api/evaluations',json={'cases':json.loads((examples/'evaluation_40.json').read_text())}).json()
                self.assertEqual(result['cases'],40)
                self.assertEqual(result['passed'],40,result)

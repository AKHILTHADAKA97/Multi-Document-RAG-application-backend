"""End-to-end tests for PDF ingestion, grounded answers and conversation memory."""
import os
import tempfile
import unittest
from pathlib import Path
os.environ['EMBEDDING_BACKEND'] = 'lexical'
from fastapi.testclient import TestClient
from app.config import Config
from app.main import create_app


def _esc(value):
    return value.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


def make_pdf(pages):
    """Build a tiny valid PDF with selectable text on each page."""
    count = len(pages)
    font = 3 + 2 * count
    kids = ' '.join(f'{3 + 2 * index} 0 R' for index in range(count))
    objects = {
        1: b'<< /Type /Catalog /Pages 2 0 R >>',
        2: f'<< /Type /Pages /Kids [{kids}] /Count {count} >>'.encode(),
        font: b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
    }
    for index, lines in enumerate(pages):
        page_id = 3 + 2 * index
        content_id = page_id + 1
        objects[page_id] = (f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] '
                            f'/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content_id} 0 R >>').encode()
        operations = ['BT', '/F1 12 Tf', '14 TL', '72 740 Td']
        for line in lines:
            operations.append(f'({_esc(line)}) Tj')
            operations.append('T*')
        operations.append('ET')
        stream = '\n'.join(operations).encode('latin-1')
        objects[content_id] = b'<< /Length ' + str(len(stream)).encode() + b' >>\nstream\n' + stream + b'\nendstream'
    output = bytearray(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offsets = {}
    for number in sorted(objects):
        offsets[number] = len(output)
        output += f'{number} 0 obj\n'.encode() + objects[number] + b'\nendobj\n'
    xref = len(output)
    output += f'xref\n0 {max(objects) + 1}\n'.encode()
    output += b'0000000000 65535 f \n'
    for number in range(1, max(objects) + 1):
        output += f'{offsets.get(number, 0):010d} 00000 n \n'.encode()
    output += f'trailer\n<< /Size {max(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode()
    return bytes(output)


PAGES = [
    ['Introduction to Artificial Intelligence',
     'Artificial Intelligence (AI) is the field of building systems that perform tasks requiring human intelligence.',
     'Machine Learning (ML) is a branch of AI that lets computers learn patterns from data without explicit programming.',
     'Deep Learning is a subset of machine learning that uses multi-layer neural networks.'],
    ['Types of Machine Learning',
     'Supervised learning trains a model on labelled examples, such as spam detection.',
     'Unsupervised learning finds structure in unlabelled data, such as customer segmentation.',
     'Reinforcement learning trains an agent using rewards and penalties.'],
    ['Examples',
     'An email spam filter learns from previously labelled emails to classify new messages as spam or legitimate.',
     'Neural networks power image recognition, speech recognition and language translation.'],
]


class UpgradeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = Config(data_dir=Path(self.temp.name), embedding_backend='lexical',
                             answer_mode='auto', app_token='')
        self.app = create_app(self.config)
        self.client = TestClient(self.app)
        upload = self.client.post('/api/documents/upload', files={'file': ('ml_guide.pdf', make_pdf(PAGES))})
        self.assertEqual(upload.status_code, 201, upload.text)
        self.document = upload.json()

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def ask(self, question, conversation=None, document_ids=None):
        payload = {'question': question, 'documentIds': document_ids or []}
        if conversation:
            payload['conversationId'] = conversation
        return self.client.post('/api/chat/ask', json=payload)

    def test_pdf_answer_contains_real_content_not_filename(self):
        response = self.ask('What is Machine Learning?').json()
        self.assertEqual(response['status'], 'answered')
        self.assertIn('branch of AI', response['answer'])
        self.assertNotEqual(response['answer'].strip(), self.document['name'])
        self.assertEqual(response['answerType'], 'document_based')
        self.assertIn(response['confidence'], {'high', 'medium'})
        page = next((s['pageNumber'] for s in response['sources'] if 'branch of AI' in s['text']), None)
        self.assertEqual(page, 1)

    def test_page_scoped_question_uses_only_that_page(self):
        response = self.ask('Explain the concept from page 2.').json()
        self.assertEqual(response['status'], 'answered')
        self.assertIn('Supervised learning', response['answer'])
        self.assertTrue(all(source['pageNumber'] == 2 for source in response['sources']))

    def test_conversation_history_resolves_follow_up_pronouns(self):
        first = self.ask('What is Machine Learning?').json()
        conversation = first['conversation_id']
        second = self.ask('What are its types?', conversation=conversation).json()
        self.assertEqual(second['status'], 'answered')
        self.assertIn('Supervised learning', second['answer'])
        third = self.ask('Explain the first type.', conversation=conversation).json()
        self.assertEqual(third['status'], 'answered')
        self.assertIn('Supervised learning', third['answer'])

    def test_summary_intent(self):
        response = self.ask('Summarize the uploaded PDF.').json()
        self.assertEqual(response['status'], 'answered')
        self.assertIn('Document summary', response['answer'])
        self.assertGreaterEqual(len(response['sources']), 2)

    def test_unrelated_question_abstains_without_sources(self):
        response = self.ask('What is quantum gravity?').json()
        self.assertEqual(response['status'], 'not_found')
        self.assertEqual(response['sources'], [])
        self.assertIn("couldn't find enough information", response['answer'])

    def test_conversation_history_and_delete_endpoints(self):
        first = self.ask('What is Machine Learning?').json()
        conversation = first['conversation_id']
        history = self.client.get(f'/api/chat/history/{conversation}')
        self.assertEqual(history.status_code, 200)
        self.assertEqual(len(history.json()['items']), 1)
        self.assertEqual(self.client.delete(f'/api/chat/{conversation}').status_code, 200)
        self.assertEqual(self.client.get(f'/api/chat/history/{conversation}').status_code, 404)

    def test_document_detail_and_invalid_document_id(self):
        detail = self.client.get('/api/documents/' + self.document['id'])
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()['name'], 'ml_guide.pdf')
        response = self.ask('What is Machine Learning?', document_ids=['missing-id'])
        self.assertEqual(response.json()['error']['code'], 'INVALID_DOCUMENT_ID')

    def test_invalid_pdf_and_scanned_pdf_messages(self):
        bad = self.client.post('/api/documents', files={'file': ('broken.pdf', b'not a pdf')})
        self.assertEqual(bad.json()['error']['code'], 'INVALID_PDF')
        from pypdf import PdfWriter
        import io
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        stream = io.BytesIO()
        writer.write(stream)
        blank = self.client.post('/api/documents/upload', files={'file': ('scan.pdf', stream.getvalue())})
        self.assertEqual(blank.json()['error']['code'], 'NO_READABLE_TEXT')

    def test_answer_formatting_is_markdown(self):
        response = self.ask('What are the types of Machine Learning?').json()
        self.assertIn('###', response['answer'])
        self.assertIn('- ', response['answer'])
        self.assertIn('[S1]', response['answer'])


if __name__ == '__main__':
    unittest.main()

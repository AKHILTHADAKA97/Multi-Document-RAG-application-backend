"""Offline tests: deterministic lexical mode; no credentials or model downloads."""
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import httpx
os.environ['EMBEDDING_BACKEND']='lexical'
from fastapi.testclient import TestClient
from app.config import Config
from app.main import create_app
from app.errors import AppError
from app.generation import generate, validate_answer
from app.parsing import extract, chunk_sections

class RAGTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.config=Config(data_dir=Path(self.temp.name),embedding_backend='lexical',answer_mode='extractive',app_token='test-secret')
        self.app=create_app(self.config)
        self.client=TestClient(self.app)
        self.headers={'X-App-Token':'test-secret'}

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def request(self,method,path,**kwargs):
        return self.client.request(method,'/api'+path,headers=self.headers,**kwargs)

    def upload(self,name='policy.txt',text=b'Employees receive 24 days of annual leave. Annual leave allowance is 24 days.'):
        return self.request('POST','/documents',files={'file':(name,text)})

    def test_auth_required_for_documents_settings_and_history(self):
        for endpoint in ['/documents','/history','/config']:
            self.assertEqual(self.client.get('/api'+endpoint).status_code,401)
        self.assertEqual(self.client.get('/health').status_code,200)

    def test_empty_workspace_question_is_logged_with_clear_error(self):
        r=self.request('POST','/chat',json={'question':'What is annual leave?'})
        self.assertEqual(r.json()['error']['code'],'NO_DOCUMENTS')
        rows=self.request('GET','/history?failed_only=true').json()
        self.assertEqual(rows['total'],1)

    def test_upload_retrieve_cite_and_delete_cascades(self):
        doc=self.upload().json()
        self.assertEqual(doc['status'],'ready')
        answer=self.request('POST','/chat',json={'question':'What is the annual leave allowance?'}).json()
        self.assertEqual(answer['status'],'answered')
        self.assertIn('24',answer['answer'])
        self.assertEqual(answer['sources'][0]['name'],'policy.txt')
        self.assertIn('[S1]',answer['answer'])
        self.request('DELETE','/documents/'+doc['id'])
        self.assertEqual(self.request('GET','/stats').json()['chunks'],0)
        self.assertEqual(self.request('GET','/history').json()['total'],1)
        self.request('DELETE','/history')
        self.assertEqual(self.request('GET','/history').json()['total'],0)

    def test_unrelated_question_abstains(self):
        self.upload()
        r=self.request('POST','/chat',json={'question':'Explain Jupiter volcanoes'}).json()
        self.assertEqual(r['status'],'not_found')
        self.assertEqual(r['sources'],[])

    def test_filter_prevents_cross_document_answer(self):
        self.upload()
        second=self.upload('cafe.txt',b'The cafe serves fresh coffee and tea.').json()
        r=self.request('POST','/chat',json={'question':'Annual leave allowance?', 'document_ids':[second['id']]}).json()
        self.assertEqual(r['status'],'not_found')

    def test_duplicate_content_not_reindexed(self):
        first=self.upload().json()
        second=self.upload('renamed.txt').json()
        self.assertEqual(first['id'],second['id'])
        self.assertTrue(second['duplicate'])
        self.assertEqual(self.request('GET','/stats').json()['documents'],1)

    def test_invalid_types_and_empty_files(self):
        self.assertEqual(self.upload('evil.exe',b'abc').json()['error']['code'],'UNSUPPORTED_FILE')
        self.assertEqual(self.upload('empty.txt',b'').json()['error']['code'],'EMPTY_FILE')
        self.assertEqual(self.upload('broken.json',b'{bad').json()['error']['code'],'PARSE_FAILED')
        self.assertEqual(self.request('GET','/documents').json()['items'][0]['status'],'failed')

    def test_upload_limit_and_capacity(self):
        self.config.max_upload_mb=1
        self.assertEqual(self.upload('large.txt',b'x'*(1024*1024+1)).status_code,413)
        self.config.max_chunks=1
        self.upload()
        self.assertEqual(self.upload('second.txt',b'Another source of unique text').json()['error']['code'],'INDEX_CAPACITY')

    def test_settings_validation_and_persistence(self):
        self.assertEqual(self.request('PUT','/settings',json={'top_k':0}).status_code,422)
        self.request('PUT','/settings',json={'top_k':3,'min_similarity':0.4})
        self.assertEqual(create_app(self.config).state.service.settings().top_k,3)

    def test_embedding_change_rejected(self):
        self.upload()
        self.app.state.service.db.set('index_signature','other-model')
        r=self.request('POST','/chat',json={'question':'Annual leave?'}).json()
        self.assertEqual(r['error']['code'],'INDEX_MISMATCH')
        r=self.upload('other.txt',b'Other unique text').json()
        self.assertEqual(r['error']['code'],'INDEX_MISMATCH')

    def test_feedback_and_evaluation(self):
        self.upload()
        r=self.request('POST','/chat',json={'question':'Annual leave allowance?'}).json()
        self.assertEqual(self.request('POST',f'/history/{r["id"]}/feedback',json={'rating':'helpful'}).status_code,200)
        result=self.request('POST','/evaluations',json={'cases':[{'question':'Annual leave allowance?', 'expected_document':'policy.txt','expected_text':'24'},{'question':'Jupiter volcanoes?', 'should_abstain':True}]}).json()
        self.assertEqual(result['passed'],2)
        self.assertEqual(result['mrr'],1)
        self.assertEqual(self.request('GET','/evaluations').json()['total'],1)

    def test_pagination_and_chunk_inspection(self):
        for i in range(12):
            self.upload(f'file{i}.txt',f'Unique file {i} for document pagination.'.encode())
        self.assertEqual(len(self.request('GET','/documents').json()['items']),10)
        page=self.request('GET','/documents?page=2').json()
        self.assertEqual(len(page['items']),2)
        r=self.request('GET',f'/documents/{page["items"][0]["id"]}/chunks').json()
        self.assertEqual(r['total'],1)

    def test_missing_key_and_model_messages(self):
        self.upload()
        self.config.answer_mode='llm'; self.config.api_key=''
        r=self.request('POST','/chat',json={'question':'Annual leave allowance?'}).json()
        self.assertEqual(r['error']['code'],'API_KEY_MISSING')
        self.config.api_key='dummy'; self.config.model=''
        r=self.request('POST','/chat',json={'question':'Annual leave allowance?'}).json()
        self.assertEqual(r['error']['code'],'MODEL_MISSING')

    def test_whitespace_question_rejected(self):
        self.assertEqual(self.request('POST','/chat',json={'question':'   '}).status_code,422)

class ParsingTests(unittest.TestCase):
    def test_csv_preserves_headers_and_row_numbers(self):
        result=extract('facts.csv',b'Policy,Value\nAnnual leave,24 days\n')
        self.assertEqual(result[0][0],'CSV row 2')
        self.assertIn('Value: 24 days',result[0][1])

    def test_docx_paragraphs_and_tables(self):
        from docx import Document
        doc=Document();doc.add_paragraph('An example paragraph')
        table=doc.add_table(rows=1,cols=2);table.cell(0,0).text='Code';table.cell(0,1).text='42'
        stream=io.BytesIO();doc.save(stream)
        result=extract('doc.docx',stream.getvalue())
        self.assertEqual(result[0][0],'Paragraph 1')
        self.assertIn('42',result[1][1])

    def test_blank_pdf_returns_ocr_guidance(self):
        from pypdf import PdfWriter
        writer=PdfWriter();writer.add_blank_page(width=100,height=100)
        stream=io.BytesIO();writer.write(stream)
        with self.assertRaises(AppError) as cm:extract('scan.pdf',stream.getvalue())
        self.assertEqual(cm.exception.code,'NO_READABLE_TEXT')

    def test_chunks_keep_overlap_and_location(self):
        chunks=chunk_sections([('Page 2',' '.join(str(i) for i in range(12)))],size=5,overlap=2)
        self.assertEqual(chunks[1]['text'],'3 4 5 6 7')
        self.assertTrue(chunks[-1]['text'].endswith('11'))
        self.assertTrue(all('Page 2' in c['location'] for c in chunks))

class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.source=[{'citation':'S1','name':'policy.txt','location':'Page 1','text':'24 days annual leave'}]
        self.temp=tempfile.TemporaryDirectory()
        self.config=Config(data_dir=Path(self.temp.name),answer_mode='llm',api_key='dummy',model='test')
    def tearDown(self):self.temp.cleanup()

    def test_unknown_or_missing_citation_rejected(self):
        for value in [{'answer':'24 [S2]','citations':['S2']},{'answer':'24','citations':['S1']}]:
            with self.assertRaises(AppError):validate_answer(value,self.source)

    def test_abstention_does_not_expose_uncited_answer(self):
        r=validate_answer({'answer':'Made up claim','citations':[]},self.source)
        self.assertEqual(r['status'],'not_found')
        self.assertNotIn('Made up',r['answer'])

    def test_provider_error_mapping(self):
        for status,code in [(401,'API_KEY_INVALID'),(403,'MODEL_ACCESS_DENIED'),(404,'MODEL_NOT_FOUND'),(429,'PROVIDER_LIMIT'),(500,'PROVIDER_UNAVAILABLE')]:
            with patch('app.generation.httpx.Client') as mock:
                mock.return_value.__enter__.return_value.post.return_value=httpx.Response(status,json={})
                with self.assertRaises(AppError) as cm:generate(self.config,'Leave?',self.source)
                self.assertEqual(cm.exception.code,code)

    def test_valid_provider_json_and_usage(self):
        with patch('app.generation.httpx.Client') as mock:
            mock.return_value.__enter__.return_value.post.return_value=httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({'answer':'24 days [S1]','citations':['S1']})}}], 'usage':{'total_tokens':100}})
            result=generate(self.config,'Leave?',self.source)
            self.assertEqual(result['status'],'answered');self.assertEqual(result['usage']['total_tokens'],100)

    def test_timeout_and_malformed_response(self):
        with patch('app.generation.httpx.Client') as mock:
            mock.return_value.__enter__.return_value.post.side_effect=httpx.ReadTimeout('timeout')
            with self.assertRaises(AppError) as cm:generate(self.config,'Leave?',self.source)
            self.assertEqual(cm.exception.code,'PROVIDER_TIMEOUT')
        with patch('app.generation.httpx.Client') as mock:
            mock.return_value.__enter__.return_value.post.return_value=httpx.Response(200,json={'choices':[]})
            with self.assertRaises(AppError) as cm:generate(self.config,'Leave?',self.source)
            self.assertEqual(cm.exception.code,'INVALID_MODEL_OUTPUT')

if __name__=='__main__':unittest.main()

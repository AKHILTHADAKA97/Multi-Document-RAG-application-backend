import json
import logging
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, Depends, Header, Query, UploadFile, File, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response, StreamingResponse
from .config import Config
from .errors import AppError
from .schemas import AskRequest, Settings, Feedback, EvalRequest, DocumentUpdate
from .service import RAGService
from .evaluation import evaluate

logging.basicConfig(level=logging.INFO)

class RateLimiter:
    """Tiny fixed-window limiter for a single-process deployment."""
    def __init__(self):
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key, limit, window=60):
        if limit <= 0:
            return True
        now = time.monotonic()
        with self.lock:
            bucket = self.hits[key]
            while bucket and now - bucket[0] > window:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True

def create_app(config=None):
    config = config or Config()
    service = RAGService(config)
    limiter = RateLimiter()
    app = FastAPI(title='Atlas Multi-Document RAG', version='1.1.0')
    app.state.service = service
    app.state.limiter = limiter
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_origin_regex=r'https://.*\.vercel\.app',
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    log = logging.getLogger('atlas.backend')
    log.info(f"Atlas Backend initialized in [{config.app_env}] mode.")
    log.info(f"Allowed CORS origins: {config.cors_origins}")
    log.info(f"Answer mode: {config.effective_answer_mode} | Embedding backend: {config.embedding_backend}")

    @app.get('/', include_in_schema=False)
    def root():
        return {
            'service': 'Atlas Multi-Document RAG API',
            'version': '1.1.0',
            'status': 'healthy',
            'environment': config.app_env,
            'docs': '/docs',
            'health': '/health',
            'ready': '/ready',
        }

    @app.get('/favicon.ico', include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @app.get('/health', include_in_schema=True)
    def health():
        return {
            'status': 'healthy',
            'environment': config.app_env,
            'answer_mode': config.effective_answer_mode,
        }

    @app.get('/health/live', include_in_schema=True)
    def health_live():
        return {
            'status': 'alive',
            'timestamp': time.time(),
            'service': 'Atlas Multi-Document RAG',
            'environment': config.app_env,
        }

    @app.get('/health/ready', include_in_schema=True)
    def health_ready():
        return ready()

    @app.get('/ready', include_in_schema=True)
    def ready():
        try:
            with service.db.connect() as con:
                con.execute('SELECT 1').fetchone()
            db_status = 'connected'
        except Exception as exc:
            logging.getLogger(__name__).error(f"Readiness check failed: {exc}")
            return JSONResponse(
                status_code=503,
                content={'status': 'not_ready', 'database': 'unavailable', 'error': 'Database unavailable'}
            )
        return {
            'status': 'ready',
            'database': db_status,
            'environment': config.app_env,
            'answer_mode': config.effective_answer_mode,
            'embedding_backend': config.embedding_backend,
        }

    @app.exception_handler(AppError)
    async def app_error(request, exc):
        return JSONResponse(status_code=exc.status, content={'error': {'code': exc.code, 'message': exc.message}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(status_code=422, content={'error': {'code': 'INVALID_INPUT', 'message': 'Check the submitted fields and limits. Questions cannot be empty.'}})

    @app.exception_handler(Exception)
    async def unexpected(request, exc):
        logging.getLogger(__name__).error('Unhandled server error', exc_info=(type(exc), exc, exc.__traceback__))
        return JSONResponse(status_code=500, content={'error': {'code': 'INTERNAL_ERROR', 'message': 'Something went wrong on the server. Check server logs or try again.'}})

    def authorize(x_app_token: str | None = Header(default=None)):
        if config.app_token and not secrets.compare_digest(x_app_token or '', config.app_token):
            raise AppError('UNAUTHORIZED', 'Enter the correct workspace access token.', 401)

    def limited(request: Request, bucket: str, limit: int):
        client = request.client.host if request.client else 'local'
        if not limiter.allow(f'{bucket}:{client}', limit):
            raise AppError('RATE_LIMITED', 'Too many requests. Wait a moment and try again.', 429)

    api = APIRouter(prefix='/api', dependencies=[Depends(authorize)])

    @api.get('/health', include_in_schema=False)
    def api_health():
        return health()

    @api.get('/health/live', include_in_schema=False)
    def api_health_live():
        return health_live()

    @api.get('/health/ready', include_in_schema=False)
    def api_health_ready():
        return health_ready()

    @api.get('/ready', include_in_schema=False)
    def api_ready():
        return ready()

    @api.get('/system/diagnostics')
    def system_diagnostics():
        return service.diagnostics()

    @api.post('/system/test-connections')
    def system_test_connections(request: Request):
        limited(request, 'system_test', 12)
        return service.test_connections()

    @api.get('/config')
    def public_config():
        return {'answer_mode': config.answer_mode, 'effective_answer_mode': config.effective_answer_mode,
                'embedding_backend': config.embedding_backend, 'embedding_model': config.embedding_model,
                'model': service.settings().model or config.model or 'not configured', 'api_key_configured': bool(config.api_key),
                'authentication_enabled': bool(config.app_token), 'max_upload_mb': config.max_upload_mb,
                'conversation_memory': True, 'rate_limit_per_minute': config.rate_limit_per_minute,
                'settings': service.settings().model_dump(),
                'available_models': [
                    {'id': 'openai/gpt-oss-120b', 'name': 'Groq GPT-OSS 120B (Advanced Reasoning)', 'provider': 'Groq (Free)'},
                    {'id': 'openai/gpt-oss-20b', 'name': 'Groq GPT-OSS 20B (Ultra Fast)', 'provider': 'Groq (Free)'},
                    {'id': 'qwen/qwen3.8-27b', 'name': 'Groq Qwen 3.8 27B', 'provider': 'Groq (Free)'},
                    {'id': 'allam-2-7b', 'name': 'Groq Allam 2 7B', 'provider': 'Groq (Free)'},
                    {'id': 'openai/gpt-oss-safeguard-20b', 'name': 'Groq GPT-OSS Safeguard 20B', 'provider': 'Groq (Free)'},
                    {'id': 'whisper-large-v3-turbo', 'name': 'Groq Whisper Large V3 Turbo', 'provider': 'Groq (Free)'},
                ]}

    @api.get('/stats')
    def stats():
        docs = service.db.rows('SELECT status,COUNT(*) AS n FROM documents GROUP BY status')
        return {'documents': sum(d['n'] for d in docs), 'ready': sum(d['n'] for d in docs if d['status'] == 'ready'),
                'chunks': service.db.rows('SELECT COUNT(*) AS n FROM chunks')[0]['n'],
                'questions': service.db.rows('SELECT COUNT(*) AS n FROM messages')[0]['n'],
                'failed_queries': service.db.rows("SELECT COUNT(*) AS n FROM messages WHERE json_extract(response,'$.status') IN ('error','not_found')")[0]['n']}

    @api.get('/documents')
    def documents(page: int = Query(1, ge=1), page_size: int = Query(10, ge=1, le=100), q: str = Query('', max_length=200)):
        args = ('%' + q + '%',)
        total = service.db.rows('SELECT COUNT(*) AS n FROM documents WHERE name LIKE ?', args)[0]['n']
        items = service.db.rows('SELECT * FROM documents WHERE name LIKE ? ORDER BY created_at DESC,id LIMIT ? OFFSET ?',
                                args + (page_size, (page - 1) * page_size))
        return {'items': items, 'total': total, 'page': page, 'page_size': page_size}

    def _save_upload(file: UploadFile):
        try:
            data = file.file.read(config.max_upload_mb * 1024 * 1024 + 1)
            return service.upload(file.filename or 'document', data)
        finally:
            file.file.close()

    @api.post('/documents', status_code=201)
    def upload(request: Request, file: UploadFile = File(...)):
        limited(request, 'upload', config.upload_rate_limit_per_minute)
        return _save_upload(file)

    @api.post('/documents/upload', status_code=201)
    def upload_alias(request: Request, file: UploadFile = File(...)):
        limited(request, 'upload', config.upload_rate_limit_per_minute)
        return _save_upload(file)

    @api.get('/documents/{document_id}')
    def document_detail(document_id: str):
        return service.document(document_id)

    @api.delete('/documents/{document_id}')
    def delete(document_id: str):
        service.delete(document_id)
        return {'deleted': True}

    @api.patch('/documents/{document_id}')
    def update_document(document_id: str, body: DocumentUpdate):
        if not service.db.rows('SELECT id FROM documents WHERE id=?', (document_id,)):
            raise AppError('DOCUMENT_NOT_FOUND', 'Document no longer exists.', 404)
        service.db.execute('UPDATE documents SET name=? WHERE id=?', (body.name, document_id))
        return {'updated': True, 'id': document_id, 'name': body.name}

    @api.post('/documents/{document_id}/retry')
    def retry_document_processing(document_id: str):
        return service.retry_document(document_id)

    @api.get('/documents/{document_id}/chunks')
    def chunks(document_id: str, page: int = Query(1, ge=1)):
        if not service.db.rows('SELECT id FROM documents WHERE id=?', (document_id,)):
            raise AppError('DOCUMENT_NOT_FOUND', 'Document no longer exists.', 404)
        items = service.db.rows('SELECT id,location,text,page_number,chunk_index FROM chunks WHERE document_id=? ORDER BY rowid LIMIT 10 OFFSET ?',
                                (document_id, (page - 1) * 10))
        total = service.db.rows('SELECT COUNT(*) AS n FROM chunks WHERE document_id=?', (document_id,))[0]['n']
        return {'items': items, 'total': total, 'page': page}

    @api.get('/documents/{document_id}/download')
    def download_document(document_id: str):
        rows = service.db.rows('SELECT * FROM documents WHERE id=?', (document_id,))
        if not rows:
            raise AppError('DOCUMENT_NOT_FOUND', 'Document no longer exists.', 404)
        doc = rows[0]
        name = doc['name']
        uploads_dir = config.data_dir / 'uploads'
        file_path = uploads_dir / f"{document_id}_{name}"
        if file_path.is_file():
            return FileResponse(file_path, filename=name)
        chunks = service.db.rows('SELECT text FROM chunks WHERE document_id=? ORDER BY rowid', (document_id,))
        text_content = "\n\n".join(c['text'] for c in chunks)
        safe_name = name if name.endswith(('.txt', '.md', '.pdf', '.docx', '.csv', '.json')) else f"{name}.txt"
        return Response(
            content=text_content,
            media_type='text/plain',
            headers={'Content-Disposition': f'attachment; filename="{safe_name}"'}
        )

    @api.post('/chat')
    def chat(request: Request, body: AskRequest):
        limited(request, 'chat', config.rate_limit_per_minute)
        accept = request.headers.get('accept', '')
        stream_param = request.query_params.get('stream', '').lower() in ('1', 'true', 'yes')
        if 'text/event-stream' in accept or stream_param:
            return StreamingResponse(
                service.answer_stream(body),
                media_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no',
                }
            )
        return service.answer(body)

    @api.post('/chat/stream')
    def chat_stream(request: Request, body: AskRequest):
        limited(request, 'chat', config.rate_limit_per_minute)
        return StreamingResponse(
            service.answer_stream(body),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
            }
        )

    @api.post('/chat/ask')
    def chat_alias(request: Request, body: AskRequest):
        limited(request, 'chat', config.rate_limit_per_minute)
        return service.answer(body)

    @api.get('/chat/history/{conversation_id}')
    def conversation_history(conversation_id: str):
        return {'conversation_id': conversation_id, 'items': service.messages_for_conversation(conversation_id)}

    @api.delete('/chat/{conversation_id}')
    def delete_conversation(conversation_id: str):
        service.delete_conversation(conversation_id)
        return {'deleted': True}

    @api.get('/history')
    def history(page: int = Query(1, ge=1), failed_only: bool = False):
        where = " WHERE json_extract(response,'$.status') IN ('error','not_found')" if failed_only else ''
        items = service.db.rows('SELECT * FROM messages' + where + ' ORDER BY created_at DESC,rowid DESC LIMIT 10 OFFSET ?',
                                ((page - 1) * 10,))
        for item in items:
            item['response'] = json.loads(item['response'])
        total = service.db.rows('SELECT COUNT(*) AS n FROM messages' + where)[0]['n']
        return {'items': items, 'total': total, 'page': page}

    @api.delete('/history/{message_id}')
    def delete_history_item(message_id: str):
        service.db.execute('DELETE FROM messages WHERE id=?', (message_id,))
        return {'deleted': True}

    @api.delete('/history')
    def clear_history():
        service.db.execute('DELETE FROM messages')
        return {'deleted': True}

    @api.post('/history/{message_id}/feedback')
    def feedback(message_id: str, body: Feedback):
        if not service.db.rows('SELECT id FROM messages WHERE id=?', (message_id,)):
            raise AppError('MESSAGE_NOT_FOUND', 'This answer is no longer in history.', 404)
        service.db.execute('INSERT OR REPLACE INTO feedback VALUES (?,?,?)', (message_id, body.rating, body.comment))
        return {'saved': True}

    @api.put('/settings')
    def settings(body: Settings):
        with service.lock:
            service.db.set('settings', body.model_dump())
        return body

    @api.post('/evaluations')
    def evaluation(body: EvalRequest):
        report = evaluate(service, body.cases)
        service.db.execute('INSERT INTO evaluations(id,report) VALUES (?,?)', (report['id'], json.dumps(report)))
        return report

    @api.get('/evaluations')
    def evaluations(page: int = Query(1, ge=1)):
        rows = service.db.rows('SELECT * FROM evaluations ORDER BY created_at DESC,rowid DESC LIMIT 10 OFFSET ?', ((page - 1) * 10,))
        return {'items': [{**r, 'report': json.loads(r['report'])} for r in rows],
                'total': service.db.rows('SELECT COUNT(*) AS n FROM evaluations')[0]['n'], 'page': page}

    app.include_router(api)
    return app

app = create_app()

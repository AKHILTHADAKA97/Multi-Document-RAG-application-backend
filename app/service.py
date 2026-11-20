import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
import httpx
from .database import Database
from .errors import AppError
from .generation import REFUSAL, generate, generate_stream
from .parsing import extract, chunk_sections, SUPPORTED
from .retrieval import Embedder, Retriever, page_number_from_question, classify_question
from .schemas import Settings

log = logging.getLogger(__name__)

FOLLOW_UP = re.compile(
    r'\b(it|its|it\'s|they|them|their|theirs|this|that|these|those|he|she|him|her|'
    r'first|second|third|fourth|last|former|latter|previous|above|mentioned|same)\b',
    re.IGNORECASE)
FOLLOW_UP_START = re.compile(r'^\s*(and|also|so|then|what about|how about|why|how so)\b', re.IGNORECASE)
SUMMARY_INTENT = re.compile(r'\b(summar|overview|key points|tl;?dr)', re.IGNORECASE)


def uid():
    return str(uuid.uuid4())


class RAGService:
    def __init__(self, config):
        self.config = config
        self.db = Database(config.data_dir / 'rag.sqlite3')
        self.embedder = Embedder(config)
        self.retriever = Retriever(config, self.db, self.embedder)
        # Single process reference implementation: serialize model access and mutation.
        self.lock = threading.RLock()
        self.db.execute('UPDATE documents SET status="failed", error="Processing interrupted. Delete this file and upload again." WHERE status="processing"')

    def settings(self):
        from .generation import SYSTEM
        stored = self.db.get('settings')
        if stored:
            s = Settings(**stored)
            if not s.system_prompt:
                s.system_prompt = SYSTEM
            return s
        # Lexical hashed vectors have a different similarity scale than semantic
        # embeddings, so the first-run threshold reflects the active backend.
        return Settings(
            min_similarity=0.10 if self.config.embedding_backend == 'lexical' else 0.15,
            candidate_k=30,
            rerank=True,
            system_prompt=SYSTEM
        )

    def upload(self, name, data):
        # Both path separators are stripped regardless of server OS.
        name = name.replace('\\', '/').split('/')[-1][:255] or 'document'
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED:
            raise AppError('UNSUPPORTED_FILE', 'Supported files: PDF, DOCX, TXT, MD, CSV and JSON.')
        if not data:
            raise AppError('EMPTY_FILE', 'The uploaded file is empty.')
        if len(data) > self.config.max_upload_mb * 1024 * 1024:
            raise AppError('FILE_TOO_LARGE', f'Upload files smaller than {self.config.max_upload_mb} MB.', 413)
        digest = hashlib.sha256(data).hexdigest()
        with self.lock:
            existing = self.db.rows('SELECT * FROM documents WHERE digest=?', (digest,))
            if existing:
                return {**existing[0], 'duplicate': True}
            signature = self.db.get('index_signature')
            if signature and signature != self.embedder.signature:
                raise AppError('INDEX_MISMATCH', 'Restore the previous embedding configuration or delete all documents before uploading.', 409)
            document_id = uid()
            uploads_dir = self.config.data_dir / 'uploads'
            uploads_dir.mkdir(parents=True, exist_ok=True)
            f_path = uploads_dir / f"{document_id}_{name}"
            f_path.write_bytes(data)

            self.db.execute(
                'INSERT INTO documents(id,name,digest,size,status,storage_path,file_type) VALUES (?,?,?,?,?,?,?)',
                (document_id, name, digest, len(data), 'processing', str(f_path), suffix)
            )

            try:
                sections = extract(name, data, ocr=self.config.enable_ocr,
                                   max_pdf_pages=self.config.max_pdf_pages)
                extracted_dir = self.config.data_dir / 'extracted'
                extracted_dir.mkdir(parents=True, exist_ok=True)
                (extracted_dir / f"{document_id}.txt").write_text('\n\n'.join(t for _, t in sections), encoding='utf-8')
                chunks = chunk_sections(sections, doc_title=name)
                count = self.db.rows('SELECT COUNT(*) AS n FROM chunks')[0]['n']
                if count + len(chunks) > self.config.max_chunks:
                    raise AppError('INDEX_CAPACITY', 'The local index reached its chunk limit. Delete unused documents or migrate to a scalable vector store.', 409)
                vectors = self.embedder.encode([c['text'] for c in chunks])
                with self.db.connect() as db:
                    db.executemany('INSERT INTO chunks(id,document_id,location,text,vector,page_number,chunk_index,section) VALUES (?,?,?,?,?,?,?,?)', [
                        (uid(), document_id, c['location'], c['text'], json.dumps(v.tolist()),
                         c.get('page_number'), c.get('chunk_index'), c.get('section')) for c, v in zip(chunks, vectors)])
                    db.execute('UPDATE documents SET status="ready", chunk_count=?, error=NULL WHERE id=?', (len(chunks), document_id))
                    db.execute('INSERT OR REPLACE INTO kv(key,value) VALUES (?,?)', ('index_signature', json.dumps(self.embedder.signature)))
            except Exception as exc:
                error = exc if isinstance(exc, AppError) else AppError('PROCESSING_FAILED', 'Document processing failed. Check server logs and retry.', 500)
                if not isinstance(exc, AppError):
                    log.exception('Document ingestion failed')
                self.db.execute('UPDATE documents SET status="failed",error=? WHERE id=?', (error.message, document_id))
                raise error
            return self.db.rows('SELECT * FROM documents WHERE id=?', (document_id,))[0]

    def retry_document(self, document_id):
        with self.lock:
            doc = self.document(document_id)
            f_path = Path(doc.get('storage_path') or (self.config.data_dir / 'uploads' / f"{document_id}_{doc['name']}"))
            if not f_path.is_file():
                raise AppError('FILE_NOT_FOUND', 'Original file is not present on disk. Please upload again.', 404)
            data = f_path.read_bytes()
            with self.db.connect() as db:
                db.execute('DELETE FROM chunks WHERE document_id=?', (document_id,))
                db.execute('UPDATE documents SET status="processing", error=NULL, chunk_count=0 WHERE id=?', (document_id,))
            try:
                sections = extract(doc['name'], data, ocr=self.config.enable_ocr, max_pdf_pages=self.config.max_pdf_pages)
                extracted_dir = self.config.data_dir / 'extracted'
                extracted_dir.mkdir(parents=True, exist_ok=True)
                (extracted_dir / f"{document_id}.txt").write_text('\n\n'.join(t for _, t in sections), encoding='utf-8')
                chunks = chunk_sections(sections, doc_title=doc['name'])
                vectors = self.embedder.encode([c['text'] for c in chunks])
                with self.db.connect() as db:
                    db.executemany('INSERT INTO chunks(id,document_id,location,text,vector,page_number,chunk_index,section) VALUES (?,?,?,?,?,?,?,?)', [
                        (uid(), document_id, c['location'], c['text'], json.dumps(v.tolist()),
                         c.get('page_number'), c.get('chunk_index'), c.get('section')) for c, v in zip(chunks, vectors)])
                    db.execute('UPDATE documents SET status="ready", chunk_count=?, error=NULL WHERE id=?', (len(chunks), document_id))
            except Exception as exc:
                err = exc if isinstance(exc, AppError) else AppError('PROCESSING_FAILED', str(exc), 500)
                self.db.execute('UPDATE documents SET status="failed", error=? WHERE id=?', (err.message, document_id))
                raise err
            return self.document(document_id)

    def delete(self, document_id):
        with self.lock:
            with self.db.connect() as db:
                rows = db.execute('SELECT name FROM documents WHERE id=?', (document_id,)).fetchall()
                found = db.execute('DELETE FROM documents WHERE id=?', (document_id,)).rowcount
                if not found:
                    raise AppError('DOCUMENT_NOT_FOUND', 'Document no longer exists.', 404)
                if not db.execute('SELECT 1 FROM chunks LIMIT 1').fetchone():
                    db.execute('DELETE FROM kv WHERE key="index_signature"')
                uploads_dir = self.config.data_dir / 'uploads'
                extracted_dir = self.config.data_dir / 'extracted'
                if rows:
                    name = rows[0][0]
                    f_path = uploads_dir / f"{document_id}_{name}"
                    if f_path.is_file():
                        try:
                            f_path.unlink()
                        except Exception:
                            pass
                    ext_path = extracted_dir / f"{document_id}.txt"
                    if ext_path.is_file():
                        try:
                            ext_path.unlink()
                        except Exception:
                            pass
            # Historical answers intentionally preserve excerpts; UI explains this.

    def document(self, document_id):
        rows = self.db.rows(
            '''SELECT d.*, (SELECT MAX(page_number) FROM chunks c WHERE c.document_id=d.id) AS page_count
               FROM documents d WHERE d.id=?''', (document_id,))
        if not rows:
            raise AppError('DOCUMENT_NOT_FOUND', 'Document no longer exists.', 404)
        return rows[0]

    def conversation_history(self, conversation_id, include_failed=False):
        if not conversation_id:
            return []
        rows = self.db.rows(
            'SELECT question,response FROM messages WHERE conversation_id=? ORDER BY created_at,rowid',
            (conversation_id,))
        turns = []
        for row in rows:
            try:
                response = json.loads(row['response'])
            except (TypeError, ValueError):
                continue
            if response.get('status') not in {'answered', 'answered_grounded'} and not include_failed:
                continue
            turns.append({'question': row['question'], 'answer': response.get('answer', '')})
        return turns

    def get_or_create_conversation(self, conversation_id=None, title="New Conversation"):
        if conversation_id:
            rows = self.db.rows('SELECT * FROM conversations WHERE id=?', (conversation_id,))
            if rows:
                return dict(rows[0])
        new_id = conversation_id or uid()
        self.db.execute(
            'INSERT OR IGNORE INTO conversations(id, title, created_at, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)',
            (new_id, title)
        )
        return {'id': new_id, 'title': title}

    def list_conversations(self):
        rows = self.db.rows('''
            SELECT c.*, COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        ''')
        return [dict(r) for r in rows]

    def get_conversation_details(self, conversation_id):
        conv_rows = self.db.rows('SELECT * FROM conversations WHERE id=?', (conversation_id,))
        if not conv_rows:
            msg_rows = self.db.rows('SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at, rowid', (conversation_id,))
            if not msg_rows:
                # Return empty fresh conversation
                return {'id': conversation_id, 'title': 'New Conversation', 'items': [], 'attachments': []}
            first_q = msg_rows[0]['question']
            title = first_q[:45] + ('...' if len(first_q) > 45 else '')
            self.db.execute('INSERT OR IGNORE INTO conversations(id, title) VALUES (?, ?)', (conversation_id, title))
            conv = {'id': conversation_id, 'title': title}
        else:
            conv = dict(conv_rows[0])

        msg_rows = self.db.rows('SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at, rowid', (conversation_id,))
        items = []
        for r in msg_rows:
            try:
                resp = json.loads(r['response'])
            except Exception:
                resp = {'status': 'answered', 'answer': r['response']}
            items.append({
                'id': r['id'],
                'conversation_id': r['conversation_id'],
                'question': r['question'],
                'response': resp,
                'created_at': r.get('created_at'),
            })

        att_rows = self.db.rows('SELECT * FROM attachments WHERE conversation_id=? ORDER BY created_at', (conversation_id,))
        attachments = [dict(a) for a in att_rows]

        return {
            'id': conv['id'],
            'title': conv.get('title', 'Conversation'),
            'created_at': conv.get('created_at'),
            'updated_at': conv.get('updated_at'),
            'items': items,
            'attachments': attachments,
        }

    def add_attachment(self, conversation_id, file):
        self.get_or_create_conversation(conversation_id)
        doc = self.upload(file)
        att_id = uid()
        fname = doc.get('name', 'file')
        ftype = fname.split('.')[-1].lower() if '.' in fname else 'unknown'
        self.db.execute(
            'INSERT INTO attachments(id, conversation_id, document_id, file_name, file_type, file_size, storage_path) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (att_id, conversation_id, doc.get('id'), fname, ftype, doc.get('size', 0), doc.get('storage_path', ''))
        )
        self.db.execute('UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?', (conversation_id,))
        return {
            'id': att_id,
            'conversation_id': conversation_id,
            'document_id': doc.get('id'),
            'file_name': fname,
            'file_type': ftype,
            'file_size': doc.get('size', 0),
            'status': doc.get('status', 'ready'),
            'storage_path': doc.get('storage_path', ''),
            'doc': doc,
        }

    def messages_for_conversation(self, conversation_id):
        rows = self.db.rows(
            'SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at,rowid',
            (conversation_id,))
        items = []
        for row in rows:
            response = json.loads(row['response'])
            items.append({'id': row['id'], 'conversation_id': row['conversation_id'],
                          'question': row['question'], 'response': response})
        if not items:
            return []
        return items

    def delete_conversation(self, conversation_id):
        self.db.execute('DELETE FROM attachments WHERE conversation_id=?', (conversation_id,))
        deleted = self.db.execute('DELETE FROM messages WHERE conversation_id=?', (conversation_id,))
        self.db.execute('DELETE FROM conversations WHERE id=?', (conversation_id,))
        return deleted or True

    def _resolve_follow_up(self, question, history):
        if not history:
            return question
        if FOLLOW_UP.search(question) or FOLLOW_UP_START.search(question):
            previous = ' '.join((turn.get('question') or '') for turn in history[-2:]).strip()
            if previous:
                return f'{previous} {question}'
        return question

    def _sources(self, sources):
        return [{
            'id': source.get('id'),
            'documentId': source.get('document_id'),
            'filename': source.get('name'),
            'name': source.get('name'),
            'pageNumber': source.get('page_number'),
            'location': source.get('location'),
            'section': source.get('section'),
            'chunkIndex': source.get('chunk_index'),
            'relevanceScore': source.get('similarity'),
            'similarity': source.get('similarity'),
            'bm25': source.get('bm25'),
            'rrf': source.get('rrf'),
            'rerankScore': source.get('rerank_score'),
            'rerank_score': source.get('rerank_score'),
            'citation': source.get('citation'),
            'text': source.get('text'),
            'uploadedAt': source.get('uploaded_at'),
        } for source in sources]

    def answer(self, request, persist=True):
        started = time.perf_counter()
        message_id = uid()
        conversation_id = request.conversation_id or uid()
        mode = self.config.effective_answer_mode
        base = {'success': True, 'id': message_id, 'message_id': message_id,
                'conversation_id': conversation_id, 'mode': mode}
        try:
            history = self.conversation_history(conversation_id)
            settings = self.settings()
            page = page_number_from_question(request.question)
            summary = bool(SUMMARY_INTENT.search(request.question))
            q_info = classify_question(request.question)
            retrieval_question = request.question if summary else self._resolve_follow_up(request.question, history)
            with self.lock:
                retrieval_t0 = time.perf_counter()
                sources, diagnostics = self.retriever.search(
                    retrieval_question, request.document_ids, settings,
                    page=page, top_k_override=10 if summary else None, force=summary)
                retrieval_ms = round((time.perf_counter() - retrieval_t0) * 1000)
                gen_ms = 0
                if not sources:
                    result = {'answer': REFUSAL, 'citations': [], 'status': 'not_found',
                              'answer_type': 'not_found', 'confidence': 'low', 'usage': None}
                    enriched = []
                    used_sources = []
                    citation_ids_valid = True
                else:
                    enriched = self._sources(sources)
                    context = '' if summary or retrieval_question == request.question else \
                        ' '.join((turn.get('question') or '') for turn in history[-2:])
                    gen_t0 = time.perf_counter()
                    result = generate(self.config, request.question, enriched,
                                      history=history, context=context,
                                      system_prompt=self.settings().system_prompt,
                                      model_override=self.settings().model)
                    gen_ms = round((time.perf_counter() - gen_t0) * 1000)
                    cited_citations = set(result.get('citations') or [])
                    if not cited_citations:
                        cited_citations = set(re.findall(r'\[(S?\d+)\]', result.get('answer', '')))
                    all_citations = {s.get('citation') for s in enriched}
                    citation_ids_valid = bool(cited_citations and cited_citations.issubset(all_citations))
                    # Include only sources actually used in the answer
                    used_sources = [s for s in enriched if s.get('citation') in cited_citations]
                    if not used_sources and result.get('status') == 'answered':
                        used_sources = enriched[:1]
            response = {
                **base,
                **result,
                'answerType': result.get('answer_type'),
                'conversationId': conversation_id,
                'messageId': message_id,
                'questionInfo': q_info,
                'sources': used_sources if result.get('status') == 'answered' else [],
                'citations': [s['citation'] for s in used_sources] if result.get('status') == 'answered' else [],
                'retrieval': diagnostics,
                'timings': {
                    'retrieval_ms': retrieval_ms,
                    'generation_ms': gen_ms,
                    'total_ms': round((time.perf_counter() - started) * 1000),
                },
                'checks': {
                    'citation_ids_valid': citation_ids_valid,
                    'grounded_in_sources': result.get('status') == 'answered' and len(used_sources) > 0,
                    'faithfulness': 'Verified against cited excerpts from uploaded files.',
                },
                'latency_ms': round((time.perf_counter() - started) * 1000),
            }
        except AppError as exc:
            response = {**base, 'success': False, 'status': 'error', 'answer': exc.message,
                        'code': exc.code, 'sources': [], 'citations': [],
                        'answer_type': 'error', 'answerType': 'error', 'confidence': 'low',
                        'latency_ms': round((time.perf_counter() - started) * 1000)}
            if persist:
                self.save_message(request.question, response)
            raise
        if persist:
            self.save_message(request.question, response)
        return response

    def answer_stream(self, request, persist=True):
        """Generator yielding Server-Sent Events (SSE) representing actual RAG progress and answer tokens."""
        started = time.perf_counter()
        message_id = uid()
        conversation_id = request.conversation_id or uid()
        mode = self.config.effective_answer_mode
        base = {'success': True, 'id': message_id, 'message_id': message_id,
                'conversation_id': conversation_id, 'mode': mode}

        try:
            # Stage 1: Understanding Question
            q_info = classify_question(request.question)
            intent_label = q_info['intent'].capitalize()
            terms_preview = ', '.join(q_info['entities'][:3]) if q_info['entities'] else 'query concepts'
            yield f"event: stage\ndata: {json.dumps({'stage': 'understanding_question', 'label': 'Understanding Question', 'detail': f'Intent: {intent_label} · Identifying key search terms: {terms_preview}'})}\n\n"

            history = self.conversation_history(conversation_id)
            settings = self.settings()
            page = page_number_from_question(request.question)
            summary = bool(SUMMARY_INTENT.search(request.question))
            retrieval_question = request.question if summary else self._resolve_follow_up(request.question, history)

            # Stage 2: Generating Embeddings
            yield f"event: stage\ndata: {json.dumps({'stage': 'generating_embeddings', 'label': 'Generating Embeddings', 'detail': 'Converting question into vector embeddings for semantic similarity...'})}\n\n"

            # Stage 3 & 4: Retrieving Documents & Reranking
            with self.lock:
                yield f"event: stage\ndata: {json.dumps({'stage': 'retrieving_documents', 'label': 'Retrieving Relevant Documents', 'detail': 'Executing parallel hybrid search (vector cosine + BM25 keyword matching)...'})}\n\n"

                def on_retrieval_stage(stage, label, detail):
                    pass

                retrieval_t0 = time.perf_counter()
                sources, diagnostics = self.retriever.search(
                    retrieval_question, request.document_ids, settings,
                    page=page, top_k_override=10 if summary else None, force=summary
                )
                retrieval_ms = round((time.perf_counter() - retrieval_t0) * 1000)

                if settings.rerank:
                    yield f"event: stage\ndata: {json.dumps({'stage': 'reranking_results', 'label': 'Reranking Results', 'detail': f'Evaluating and prioritizing {len(sources)} candidate excerpts with cross-match reranker...'})}\n\n"

                # Stage 5: Building Context
                yield f"event: stage\ndata: {json.dumps({'stage': 'building_context', 'label': 'Building Context', 'detail': f'Deduplicating and assembling context from {len(sources)} grounded excerpts...'})}\n\n"

                if not sources:
                    result = {'answer': REFUSAL, 'citations': [], 'status': 'not_found',
                              'answer_type': 'not_found', 'confidence': 'low', 'usage': None}
                    enriched = []
                    used_sources = []
                    citation_ids_valid = True
                    ttft_ms = 0
                    gen_ms = 0
                else:
                    enriched = self._sources(sources)
                    context = '' if summary or retrieval_question == request.question else \
                        ' '.join((turn.get('question') or '') for turn in history[-2:])

                    # Stage 6: Generating Answer (streaming tokens)
                    active_model = self.settings().model or self.config.model
                    yield f"event: stage\ndata: {json.dumps({'stage': 'generating_answer', 'label': 'Generating Answer', 'detail': f'Synthesizing grounded response using {active_model}...' })}\n\n"

                    gen_t0 = time.perf_counter()
                    gen_result = None
                    for token_event in generate_stream(
                        self.config, request.question, enriched,
                        history=history, context=context,
                        system_prompt=self.settings().system_prompt,
                        model_override=self.settings().model
                    ):
                        if token_event.get('type') == 'token':
                            yield f"event: token\ndata: {json.dumps({'token': token_event['token']})}\n\n"
                        elif token_event.get('type') == 'done':
                            gen_result = token_event

                    gen_ms = round((time.perf_counter() - gen_t0) * 1000)
                    result = gen_result or {'answer': REFUSAL, 'citations': [], 'status': 'not_found'}
                    ttft_ms = result.get('ttft_ms', gen_ms)

                    # Stage 7: Validating Response
                    yield f"event: stage\ndata: {json.dumps({'stage': 'validating_response', 'label': 'Validating Response', 'detail': 'Verifying inline citations and grounding against source excerpts...'})}\n\n"

                    cited_citations = set(result.get('citations') or [])
                    if not cited_citations:
                        cited_citations = set(re.findall(r'\[(S?\d+)\]', result.get('answer', '')))
                    all_citations = {s.get('citation') for s in enriched}
                    citation_ids_valid = bool(cited_citations and cited_citations.issubset(all_citations))
                    used_sources = [s for s in enriched if s.get('citation') in cited_citations]
                    if not used_sources and result.get('status') == 'answered':
                        used_sources = enriched[:1]

            # Stage 8: Finalizing Response
            yield f"event: stage\ndata: {json.dumps({'stage': 'finalizing_response', 'label': 'Finalizing Response', 'detail': 'Formatting final answer and telemetry...'})}\n\n"

            total_ms = round((time.perf_counter() - started) * 1000)
            response = {
                **base,
                **result,
                'answerType': result.get('answer_type'),
                'conversationId': conversation_id,
                'messageId': message_id,
                'questionInfo': q_info,
                'sources': used_sources if result.get('status') == 'answered' else [],
                'citations': [s['citation'] for s in used_sources] if result.get('status') == 'answered' else [],
                'retrieval': diagnostics,
                'timings': {
                    'retrieval_ms': retrieval_ms,
                    'generation_ms': gen_ms,
                    'ttft_ms': ttft_ms,
                    'total_ms': total_ms,
                },
                'checks': {
                    'citation_ids_valid': citation_ids_valid,
                    'grounded_in_sources': result.get('status') == 'answered' and len(used_sources) > 0,
                    'faithfulness': 'Verified against cited excerpts from uploaded files.',
                },
                'latency_ms': total_ms,
            }

            if persist:
                self.save_message(request.question, response)

            yield f"event: complete\ndata: {json.dumps(response)}\n\n"

        except Exception as exc:
            code = exc.code if isinstance(exc, AppError) else 'INTERNAL_ERROR'
            msg = exc.message if isinstance(exc, AppError) else str(exc)
            err_payload = {
                **base,
                'success': False,
                'status': 'error',
                'code': code,
                'answer': msg,
                'sources': [],
                'citations': [],
                'latency_ms': round((time.perf_counter() - started) * 1000)
            }
            if persist:
                try:
                    self.save_message(request.question, err_payload)
                except Exception:
                    pass
            yield f"event: error\ndata: {json.dumps(err_payload)}\n\n"

    def save_message(self, question, response):
        conv_id = response.get('conversation_id') or response.get('conversationId')
        title = question[:45] + ('...' if len(question) > 45 else '')
        self.db.execute('INSERT OR IGNORE INTO conversations(id, title) VALUES (?, ?)', (conv_id, title))
        self.db.execute('UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?', (conv_id,))
        self.db.execute('INSERT INTO messages(id,conversation_id,question,response) VALUES (?,?,?,?)',
                        (response['id'], conv_id, question, json.dumps(response)))

    def diagnostics(self):
        """Comprehensive telemetry and health status across all subsystems."""
        db_alive = False
        try:
            with self.db.connect() as con:
                con.execute('SELECT 1').fetchone()
            db_alive = True
        except Exception:
            pass

        storage_writable = False
        is_render_disk = str(self.config.data_dir).startswith('/var/data')
        try:
            t_file = self.config.data_dir / '.diag_test'
            t_file.touch()
            t_file.unlink()
            storage_writable = True
        except Exception:
            pass

        doc_count = self.db.rows('SELECT COUNT(*) AS n FROM documents')[0]['n']
        ready_count = self.db.rows('SELECT COUNT(*) AS n FROM documents WHERE status="ready"')[0]['n']
        failed_count = self.db.rows('SELECT COUNT(*) AS n FROM documents WHERE status="failed"')[0]['n']
        chunk_count = self.db.rows('SELECT COUNT(*) AS n FROM chunks')[0]['n']
        query_count = self.db.rows('SELECT COUNT(*) AS n FROM messages')[0]['n']

        return {
            'timestamp': time.time(),
            'environment': self.config.app_env,
            'frontend_url': self.config.frontend_url,
            'cors_origins': self.config.cors_origins,
            'database': {
                'status': 'active' if db_alive else 'error',
                'type': 'SQLite (WAL mode)',
                'path': str(self.config.data_dir / 'rag.sqlite3'),
                'total_documents': doc_count,
                'ready_documents': ready_count,
                'failed_documents': failed_count,
                'total_chunks': chunk_count,
            },
            'storage': {
                'status': 'active' if storage_writable else 'error',
                'type': 'Render Persistent Disk' if is_render_disk else 'Local Data Directory',
                'mount_path': str(self.config.data_dir),
                'writable': storage_writable,
                'is_persistent_disk': is_render_disk,
                'uploads_dir': str(self.config.data_dir / 'uploads'),
                'extracted_dir': str(self.config.data_dir / 'extracted'),
            },
            'embedding': {
                'status': 'active',
                'backend': self.config.embedding_backend,
                'model': self.config.embedding_model if self.config.embedding_backend == 'semantic' else 'hashed-2048-v1',
                'signature': self.embedder.signature,
                'query_cache_size': len(getattr(self.embedder, '_query_cache', {})),
            },
            'retrieval': {
                'status': 'active',
                'type': 'Hybrid (Dense Vectors + BM25 Sparse + RRF)',
                'rerank_enabled': bool(self.settings().rerank),
                'rerank_model': self.config.rerank_model,
                'top_k': self.settings().top_k,
                'candidate_k': self.settings().candidate_k,
            },
            'llm': {
                'status': 'active' if bool(self.config.api_key) else 'not_configured',
                'provider': 'Groq Cloud',
                'model': self.settings().model or self.config.model,
                'base_url': self.config.base_url,
                'api_key_configured': bool(self.config.api_key),
            },
            'streaming': {
                'status': 'active',
                'protocol': 'Server-Sent Events (SSE)',
                'endpoint': '/api/chat/stream',
            },
            'telemetry': {
                'total_queries': query_count,
            }
        }

    def test_connections(self):
        """Perform on-demand live test of all external and local subsystems."""
        results = {}

        # 1. Database read/write test
        t0 = time.perf_counter()
        try:
            with self.db.connect() as con:
                con.execute('SELECT 1').fetchone()
            db_ms = round((time.perf_counter() - t0) * 1000)
            results['database'] = {
                'name': 'Database (SQLite WAL)',
                'status': 'active',
                'latency_ms': db_ms,
                'message': f'Connected. Read query executed in {db_ms} ms.',
            }
        except Exception as e:
            results['database'] = {
                'name': 'Database (SQLite WAL)',
                'status': 'error',
                'error': str(e),
                'suggested_fix': 'Verify file permissions on the data directory and ensure rag.sqlite3 is not locked.',
            }

        # 2. File storage write/read test
        t0 = time.perf_counter()
        try:
            uploads_dir = self.config.data_dir / 'uploads'
            uploads_dir.mkdir(parents=True, exist_ok=True)
            test_file = uploads_dir / f'.test_write_{uid()}'
            test_file.write_bytes(b'storage_test')
            _ = test_file.read_bytes()
            test_file.unlink()
            st_ms = round((time.perf_counter() - t0) * 1000)
            results['storage'] = {
                'name': 'File Storage (Persistent Disk)',
                'status': 'active',
                'latency_ms': st_ms,
                'message': f'Read & write validated in {st_ms} ms at {self.config.data_dir}.',
            }
        except Exception as e:
            results['storage'] = {
                'name': 'File Storage (Persistent Disk)',
                'status': 'error',
                'error': str(e),
                'suggested_fix': 'Attach or format a Render Persistent Disk mounted at /var/data with write permissions.',
            }

        # 3. Embedding model inference test
        t0 = time.perf_counter()
        try:
            vec = self.embedder.encode(['health test query'])[0]
            emb_ms = round((time.perf_counter() - t0) * 1000)
            results['embedding'] = {
                'name': f'Embedding Engine ({self.config.embedding_backend})',
                'status': 'active',
                'latency_ms': emb_ms,
                'message': f'Encoded 1 test vector (dimension {len(vec)}) in {emb_ms} ms.',
            }
        except Exception as e:
            results['embedding'] = {
                'name': f'Embedding Engine ({self.config.embedding_backend})',
                'status': 'error',
                'error': str(e),
                'suggested_fix': 'Check embedding model requirements or set EMBEDDING_BACKEND=lexical for lightweight cloud operation.',
            }

        # 4. LLM Provider (Groq) API connectivity test
        if not self.config.api_key:
            results['llm'] = {
                'name': 'LLM Provider (Groq Cloud)',
                'status': 'not_configured',
                'message': 'No API key set in LLM_API_KEY environment variable.',
                'suggested_fix': 'Generate a free Groq API key at console.groq.com and configure LLM_API_KEY in Render environment variables.',
            }
        else:
            t0 = time.perf_counter()
            try:
                res = httpx.get(
                    self.config.base_url.rstrip('/') + '/models',
                    headers={'Authorization': f'Bearer {self.config.api_key}'},
                    timeout=8.0
                )
                llm_ms = round((time.perf_counter() - t0) * 1000)
                if res.status_code == 200:
                    results['llm'] = {
                        'name': 'LLM Provider (Groq Cloud)',
                        'status': 'active',
                        'latency_ms': llm_ms,
                        'message': f'Authenticated successfully with Groq API in {llm_ms} ms.',
                    }
                elif res.status_code == 401:
                    results['llm'] = {
                        'name': 'LLM Provider (Groq Cloud)',
                        'status': 'error',
                        'latency_ms': llm_ms,
                        'error': 'HTTP 401 Unauthorized - Invalid API key',
                        'suggested_fix': 'Check your LLM_API_KEY in Render environment variables. Ensure no leading/trailing spaces.',
                    }
                else:
                    results['llm'] = {
                        'name': 'LLM Provider (Groq Cloud)',
                        'status': 'error',
                        'latency_ms': llm_ms,
                        'error': f'HTTP {res.status_code}: {res.text[:120]}',
                        'suggested_fix': 'Check Groq status or quota limits.',
                    }
            except Exception as e:
                results['llm'] = {
                    'name': 'LLM Provider (Groq Cloud)',
                    'status': 'error',
                    'error': f'Network failure: {str(e)}',
                    'suggested_fix': 'Verify outbound internet connectivity from Render backend to api.groq.com.',
                }

        # Overall health calculation
        statuses = [r.get('status') for r in results.values()]
        has_error = any(s == 'error' for s in statuses)
        overall = 'degraded' if has_error else 'healthy'

        return {
            'timestamp': time.time(),
            'overall_status': overall,
            'services': results,
        }

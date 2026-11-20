"""SQLite metadata + vector persistence. All document/chunk writes are atomic."""
import json
import sqlite3
from contextlib import contextmanager

class Database:
    def __init__(self, path):
        self.path = str(path)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
              id TEXT PRIMARY KEY, name TEXT NOT NULL, digest TEXT UNIQUE NOT NULL,
              size INTEGER NOT NULL, status TEXT NOT NULL, error TEXT,
              chunk_count INTEGER NOT NULL DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
              storage_path TEXT, file_type TEXT);
            CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
              location TEXT NOT NULL, text TEXT NOT NULL, vector TEXT NOT NULL,
              page_number INTEGER, chunk_index INTEGER, section TEXT);
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
            CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages (
              id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, question TEXT NOT NULL,
              response TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
            CREATE TABLE IF NOT EXISTS feedback (
              message_id TEXT PRIMARY KEY REFERENCES messages(id) ON DELETE CASCADE,
              rating TEXT NOT NULL, comment TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evaluations (
              id TEXT PRIMARY KEY, report TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            """)
            self._migrate(db)

    @staticmethod
    def _migrate(db):
        """Add columns introduced after an older database file was created."""
        columns = {row[1] for row in db.execute('PRAGMA table_info(chunks)')}
        if 'page_number' not in columns:
            db.execute('ALTER TABLE chunks ADD COLUMN page_number INTEGER')
        if 'chunk_index' not in columns:
            db.execute('ALTER TABLE chunks ADD COLUMN chunk_index INTEGER')
        if 'section' not in columns:
            db.execute('ALTER TABLE chunks ADD COLUMN section TEXT')

        doc_cols = {row[1] for row in db.execute('PRAGMA table_info(documents)')}
        if 'storage_path' not in doc_cols:
            db.execute('ALTER TABLE documents ADD COLUMN storage_path TEXT')
        if 'file_type' not in doc_cols:
            db.execute('ALTER TABLE documents ADD COLUMN file_type TEXT')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def rows(self, sql, args=()):
        with self.connect() as db:
            return [dict(row) for row in db.execute(sql, args).fetchall()]

    def execute(self, sql, args=()):
        with self.connect() as db:
            return db.execute(sql, args).rowcount

    def get(self, key, default=None):
        rows = self.rows('SELECT value FROM kv WHERE key=?', (key,))
        return json.loads(rows[0]['value']) if rows else default

    def set(self, key, value):
        self.execute('INSERT OR REPLACE INTO kv(key,value) VALUES (?,?)', (key, json.dumps(value)))

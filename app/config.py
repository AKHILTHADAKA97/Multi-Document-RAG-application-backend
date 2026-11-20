"""Process configuration. Secrets stay on the server; restart after .env edits."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / '.env')
load_dotenv()

def _parse_origins() -> list[str]:
    defaults = {
        'https://muldocrag.vercel.app',
        'http://localhost:5173',
        'http://127.0.0.1:5173',
        'http://localhost:3000',
        'http://127.0.0.1:3000',
    }
    raw = os.getenv('CORS_ORIGINS', '')
    if raw:
        for item in raw.split(','):
            cleaned = item.strip().rstrip('/')
            if cleaned:
                defaults.add(cleaned)
    return list(defaults)

@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv('DATA_DIR', './data')))
    embedding_backend: str = field(default_factory=lambda: os.getenv('EMBEDDING_BACKEND', 'semantic'))
    embedding_model: str = field(default_factory=lambda: os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
    rerank_model: str = field(default_factory=lambda: os.getenv('RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2'))
    # auto: use the configured chat model when a key and model exist, otherwise
    # generate a grounded answer locally from retrieved passages.
    answer_mode: str = field(default_factory=lambda: os.getenv('ANSWER_MODE', 'auto'))
    base_url: str = field(default_factory=lambda: os.getenv('LLM_BASE_URL') or os.getenv('GROQ_BASE_URL', 'https://api.groq.com/openai/v1'))
    model: str = field(default_factory=lambda: os.getenv('LLM_MODEL') or os.getenv('GROQ_MODEL', 'openai/gpt-oss-120b'))
    api_key: str = field(default_factory=lambda: os.getenv('LLM_API_KEY') or os.getenv('GROQ_API_KEY', ''))
    app_token: str = field(default_factory=lambda: os.getenv('APP_TOKEN', ''))
    cors_origins: list[str] = field(default_factory=_parse_origins)
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv('MAX_UPLOAD_MB', '20')))
    max_chunks: int = field(default_factory=lambda: int(os.getenv('MAX_CHUNKS', '10000')))
    max_pdf_pages: int = field(default_factory=lambda: int(os.getenv('MAX_PDF_PAGES', '500')))
    max_context_chars: int = field(default_factory=lambda: int(os.getenv('MAX_CONTEXT_CHARS', '14000')))
    history_turns: int = field(default_factory=lambda: int(os.getenv('HISTORY_TURNS', '6')))
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.getenv('RATE_LIMIT_PER_MINUTE', '120')))
    upload_rate_limit_per_minute: int = field(default_factory=lambda: int(os.getenv('UPLOAD_RATE_LIMIT_PER_MINUTE', '30')))
    enable_ocr: bool = field(default_factory=lambda: os.getenv('ENABLE_OCR', 'false').lower() == 'true')

    def __post_init__(self):
        if self.embedding_backend not in {'semantic', 'lexical'}:
            raise ValueError('EMBEDDING_BACKEND must be semantic or lexical')
        if self.answer_mode not in {'auto', 'extractive', 'llm'}:
            raise ValueError('ANSWER_MODE must be auto, extractive or llm')
        if self.max_chunks < 1 or self.max_upload_mb < 1:
            raise ValueError('Upload and chunk limits must be positive')
        if self.max_pdf_pages < 1 or self.max_context_chars < 1000:
            raise ValueError('Page and context limits must be positive')
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def effective_answer_mode(self):
        """Resolve auto mode without hiding an explicitly configured mode."""
        if self.answer_mode == 'llm':
            return 'llm'
        if self.answer_mode == 'auto' and self.api_key and self.model:
            return 'llm'
        return 'extractive'

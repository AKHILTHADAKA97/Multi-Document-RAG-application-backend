"""Extract clean, located text from supported files.

PDF page numbers are preserved because citations depend on them. Chunking is
word-window based but snaps to sentence boundaries when possible so headings,
paragraphs and list items are not cut in half.
"""
import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from pathlib import Path
from .errors import AppError

IMAGE_SUPPORTED = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
SUPPORTED = {'.pdf', '.docx', '.txt', '.md', '.csv', '.json'} | IMAGE_SUPPORTED
MIN_MEANINGFUL_CHARS = 3


def _meaningful(text: str) -> bool:
    return len(re.findall(r'[A-Za-z0-9]', text)) >= MIN_MEANINGFUL_CHARS


def _mark_headings(text: str) -> str:
    """Give short, punctuation-free heading lines a terminator so chunk text
    can later be split back into sentences."""
    lines = text.split('\n')
    for index, line in enumerate(lines):
        stripped = line.strip()
        words = stripped.split()
        if (stripped and 1 <= len(words) <= 10 and index < len(lines) - 1
                and not re.search(r'[.!?:;,]$', stripped)
                and re.match(r'^[A-Z0-9]', stripped)):
            lines[index] = stripped + '.'
    return '\n'.join(lines)


def clean_text(text: str) -> str:
    """Normalise extracted text without destroying sentence boundaries."""
    if not text:
        return ''
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\ufeff', '').replace('\u00ad', '')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Re-join words broken by a hyphen at the end of a line.
    text = re.sub(r'(?<=[A-Za-z])-\s*\n\s*(?=[a-z])', '', text)
    # Common PDF ligatures and typographic spaces.
    text = text.replace('\ufb01', 'fi').replace('\ufb02', 'fl')
    text = text.replace('\u00a0', ' ').replace('\u2007', ' ').replace('\u202f', ' ')
    text = re.sub(r'[ \t\f\v]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return _mark_headings(text.strip())


def _ocr_pdf(data: bytes):
    """Optional OCR path. Requires pdf2image and pytesseract plus Tesseract."""
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
    except ImportError as exc:
        raise AppError(
            'OCR_UNAVAILABLE',
            'This PDF has no extractable text. Install OCR support (pdf2image and pytesseract) or upload a searchable PDF.',
        ) from exc
    try:
        images = convert_from_bytes(data)
        return [(f'Page {i + 1}', clean_text(pytesseract.image_to_string(image))) for i, image in enumerate(images)]
    except Exception as exc:
        raise AppError(
            'OCR_FAILED',
            'OCR could not read this PDF. Install the Tesseract binary or upload a searchable PDF.',
        ) from exc


def extract(name: str, data: bytes, ocr: bool = False, max_pdf_pages: int = 500):
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED:
        raise AppError('UNSUPPORTED_FILE', 'Supported files: PDF, DOCX, TXT, MD, CSV and JSON.')
    try:
        if suffix == '.pdf':
            is_real_pdf = b'%PDF' in data[:1024]
            sections = []
            if not is_real_pdf:
                # The file is not a real binary PDF (e.g. plain text with a .pdf extension)
                try:
                    text = clean_text(data.decode('utf-8-sig'))
                    if _meaningful(text):
                        sections = [('Text section', text)]
                except UnicodeDecodeError:
                    raise AppError('INVALID_PDF', 'This file is not a valid PDF document.')
            else:
                from pypdf import PdfReader
                try:
                    reader = PdfReader(io.BytesIO(data))
                    if reader.is_encrypted:
                        try:
                            reader.decrypt('')
                        except Exception:
                            pass
                        if reader.is_encrypted:
                            raise AppError('ENCRYPTED_PDF', 'Remove the PDF password and upload it again.')
                    if len(reader.pages) > max_pdf_pages:
                        raise AppError('PDF_TOO_LONG', f'PDFs are limited to {max_pdf_pages} pages.', 413)
                    for i, page in enumerate(reader.pages):
                        text = clean_text(page.extract_text() or '')
                        if _meaningful(text):
                            sections.append((f'Page {i + 1}', text))
                except AppError:
                    raise
                except Exception:
                    try:
                        text = clean_text(data.decode('utf-8-sig'))
                        if _meaningful(text):
                            sections = [('Text section', text)]
                    except Exception:
                        pass
            if not sections:
                if ocr and is_real_pdf:
                    sections = [(location, text) for location, text in _ocr_pdf(data) if _meaningful(text)]
                if not sections:
                    raise AppError(
                        'NO_READABLE_TEXT',
                        'No readable text was found. This PDF may be scanned or image-only; run OCR first and upload the searchable PDF.',
                    )
        elif suffix == '.docx':
            # Bound expanded archive size before python-docx reads the document.
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if sum(item.file_size for item in archive.infolist()) > 80 * 1024 * 1024:
                    raise AppError('EXPANDED_FILE_TOO_LARGE', 'This DOCX expands beyond the 80 MB limit.')
            from docx import Document
            doc = Document(io.BytesIO(data))
            sections = [(f'Paragraph {i + 1}', clean_text(p.text)) for i, p in enumerate(doc.paragraphs)]
            for i, table in enumerate(doc.tables):
                rows = '\n'.join(' | '.join(c.text for c in row.cells) for row in table.rows)
                sections.append((f'Table {i + 1}', clean_text(rows)))
        elif suffix in IMAGE_SUPPORTED:
            w, h = 0, 0
            if suffix == '.png' and len(data) >= 24 and data[:8] == b'\x89PNG\r\n\x1a\n':
                w = int.from_bytes(data[16:20], 'big')
                h = int.from_bytes(data[20:24], 'big')
            elif suffix in {'.jpg', '.jpeg'} and len(data) >= 2:
                idx = 2
                while idx < len(data) - 9:
                    if data[idx] == 0xFF and data[idx+1] in (0xC0, 0xC1, 0xC2):
                        h = int.from_bytes(data[idx+5:idx+7], 'big')
                        w = int.from_bytes(data[idx+7:idx+9], 'big')
                        break
                    elif data[idx] == 0xFF:
                        length = int.from_bytes(data[idx+2:idx+4], 'big')
                        idx += 2 + length
                    else:
                        idx += 1
            size_kb = round(len(data) / 1024, 1)
            dim_str = f"{w}x{h}" if w and h else "Standard resolution"
            clean_name = Path(name).stem.replace('_', ' ').replace('-', ' ')
            content = f"Image Attachment: {name}. Topic: {clean_name}. Format: {suffix.upper()[1:]}. Dimensions: {dim_str}. File size: {size_kb} KB. Attached to conversation for analysis and discussion."
            sections = [('Image Overview', content)]
        else:
            text = data.decode('utf-8-sig')
            if suffix == '.json':
                value = json.loads(text)
                sections = [('JSON document', clean_text(json.dumps(value, ensure_ascii=False, indent=2)))]
            elif suffix == '.csv':
                rows = list(csv.reader(io.StringIO(text)))
                header = rows[0] if rows else []
                sections = [(f'CSV row {i + 2}', clean_text(' | '.join(
                    f'{header[j] if j < len(header) else j + 1}: {v}' for j, v in enumerate(row))))
                    for i, row in enumerate(rows[1:])]
            else:
                sections = [('Text section', clean_text(text))]
        sections = [(location, text) for location, text in sections if text.strip() and _meaningful(text)]
        if not sections:
            raise AppError(
                'NO_READABLE_TEXT',
                'No readable text found. Scan the file again or upload a document that contains selectable text.',
            )
        return sections
    except AppError:
        raise
    except UnicodeDecodeError:
        raise AppError('ENCODING_ERROR', 'Save text, CSV or JSON as UTF-8 and upload again.')
    except Exception as exc:
        raise AppError('PARSE_FAILED', 'The file could not be read. Check that it is valid and not corrupted.') from exc


def _page_number(location: str):
    match = re.search(r'Page (\d+)', location)
    return int(match.group(1)) if match else None


def _chunk_words(words, size, overlap):
    """Window of words that prefers ending on sentence punctuation."""
    step = size - overlap
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        if end < len(words):
            floor = start + max(1, size // 2)
            for index in range(end - 1, floor - 1, -1):
                if words[index][-1:] in '.!?:;':
                    end = index + 1
                    break
        yield start, end, ' '.join(words[start:end])
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)


def split_into_sections(text: str) -> list[str]:
    """Split before each numbered topic heading, e.g. '2. What is ANN ...'."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.?!])\s+(?=\d{1,2}\.\s+[A-Z])", text)
    if len(parts) <= 1:
        # Also try newline-separated headings or markdown headings
        parts = re.split(r"(?:\r?\n)+(?=(?:#{1,4}\s+|\d{1,2}\.\s+[A-Z]))", text)
    return [p.strip() for p in parts if len(p.strip()) >= 30]


def dedupe_chunks(chunks: list[dict]) -> list[dict]:
    """Deduplicate chunks by alphanumeric normalized content hash."""
    seen, unique = set(), []
    for c in chunks:
        key = hashlib.md5(re.sub(r"\W+", "", c['text'].lower()).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def chunk_sections(sections, size=200, overlap=40, doc_title=""):
    """Chunk text into section/topic-based passages with duplicates removed.

    If the text has numbered headings or distinct sections, splits by section
    so concepts are not intermingled. Otherwise, chunks by word window snapping
    to sentence boundaries.
    """
    raw_chunks = []
    chunk_counter = 1

    for location, text in sections:
        page = _page_number(location)
        sub_sections = split_into_sections(text)
        if len(sub_sections) > 1:
            for sub in sub_sections:
                heading = re.split(r"[:?]", sub, maxsplit=1)[0][:60].strip()
                words = sub.split()
                if len(words) > 350:
                    for num, (_, _, chunk_text) in enumerate(_chunk_words(words, size=size, overlap=overlap), 1):
                        loc = f"Page {page} · {heading}" if page is not None else f'"{heading}", page N/A'
                        raw_chunks.append({
                            'location': loc,
                            'section': heading,
                            'text': chunk_text,
                            'page_number': page,
                            'chunk_index': chunk_counter,
                        })
                        chunk_counter += 1
                else:
                    loc = f"Page {page} · {heading}" if page is not None else f'"{heading}", page N/A'
                    raw_chunks.append({
                        'location': loc,
                        'section': heading,
                        'text': sub,
                        'page_number': page,
                        'chunk_index': chunk_counter,
                    })
                    chunk_counter += 1
        else:
            words = text.split()
            if not words:
                continue
            heading = re.split(r"[:?]", text, maxsplit=1)[0][:60].strip() if len(text) > 20 else location
            for number, (_, _, chunk_text) in enumerate(_chunk_words(words, size=size, overlap=overlap), 1):
                loc = f"{location} · chunk {number}" if page is not None else f'"{heading}", page N/A'
                raw_chunks.append({
                    'location': loc,
                    'section': heading,
                    'text': chunk_text,
                    'page_number': page,
                    'chunk_index': chunk_counter,
                })
                chunk_counter += 1

    # Remove duplicates
    unique_chunks = dedupe_chunks(raw_chunks)
    for idx, c in enumerate(unique_chunks, 1):
        c['chunk_index'] = idx
    return unique_chunks

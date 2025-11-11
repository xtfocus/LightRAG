## Modular File Processor Plan

### Goal
Let each teammate own a file-type extractor while the FastAPI upload flow stays unchanged.

### Architecture Snapshot
- `document_routes.py` keeps handling uploads, storage, and queueing.
- New package `lightrag/api/document_processors/` contains one module per file type.
- A registry in `document_processors/base.py` maps extensions to processors.
- When the API starts, it imports the processor modules so they register themselves.
- `pipeline_enqueue_file` reads the file, looks up the processor by extension, runs it, and hands the returned text chunks to `LightRAG.apipeline_enqueue_documents`.

### Processor Contract
- Module location: `lightrag/api/document_processors/<ext>_processor.py`
- Register exactly one handler per extension using `register(".ext", processor)`.
- Function signature:

```0:18:lightrag/api/document_processors/template_processor.py
@register(".ext")
def process_ext(data: bytes, filename: str) -> list[str]:
    """
    Args:
        data: raw bytes read from the uploaded file.
        filename: sanitized basename (including extension) for logging and context.
    Returns:
        list[str]: human-readable text chunks ready for ingestion.
    Raises:
        ValueError: for recoverable issues (bad encoding, unsupported layout, passwords, etc.).
    """
```

- Input details:
  - `data` is exactly what the router read from disk—no additional IO inside the processor.
  - `filename` is safe to log and helps identify failures.
- Output requirements:
  - Return `list[str]`. No other container types are accepted.
  - Each element must be textual and respect the shared chunk-size limit (`MAX_CHUNK_CHARS`, currently 2000). Use the shared `chunk_text` helper or manual slicing.
  - Return an empty list only if nothing can be extracted; otherwise raise `ValueError`.
- Error handling:
  - Raise `ValueError` when the format cannot be parsed or is unsupported.
  - Unexpected exceptions propagate so the router can log them.

### Integration Touchpoints
1. `document_routes.py` calls `load_processors()` (bootstrap import) during router setup.
2. For each uploaded file:
   - `ext = file_path.suffix.lower()`
   - `processor = get_processor(ext)`
   - If no processor exists, reuse the current “unsupported file type” path.
   - Otherwise run `chunks = processor(file_bytes, file_path.name)` and feed the returned `list[str]` to `rag.apipeline_enqueue_documents`.
3. Keep all existing logging, queue management, and `__enqueued__` file moves unchanged.

### Example Ownership Table
| Extension | Owner | Module              | Quick note                   |
|-----------|-------|---------------------|------------------------------|
| `.docx`   | Alice | `docx_processor.py` | Use `python-docx`.           |
| `.pdf`    | Bob   | `pdf_processor.py`  | Use `pypdf`, handle password |
| `.md`     | Carol | `md_processor.py`   | Normalize front matter.      |

Update responsibilities here as they shift.


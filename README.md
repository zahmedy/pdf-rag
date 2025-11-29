# PDF RAG Assistant
A tiny, self-contained RAG CLI that lets you chat with the PDFs in `data/docs`.

## Prerequisites
- Python 3.10+
- An OpenAI API key (`OPENAI_API_KEY`)

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

echo "OPENAI_API_KEY=sk-..." > .env
python main.py --pdf-dir data/docs
```

## Usage
- Interactive chat: `python main.py`
- Ask one question: `python main.py --question "What is this document about?"`
- Force rebuild the index: `python main.py --rebuild`
- Custom locations: `--pdf-dir path/to/pdfs --index-dir path/to/faiss_index`

The first run loads your PDFs, chunks them, and builds a FAISS index (saved under `faiss_index/`). Subsequent runs reuse the saved index unless you pass `--rebuild`.

## Project layout
```
data/
  docs/              # Drop your PDFs here
faiss_index/         # Auto-created FAISS index (gitignored)
main.py              # CLI entry point
utils.py             # RAG helpers (load/split/index/query)
requirements.txt     # Python deps
.env                 # OPENAI_API_KEY (not committed)
```

## Notes
- Sources are printed with every answer so you can see which PDF pages were used.
- If you change the PDFs, pass `--rebuild` to refresh the index.

import argparse
from pathlib import Path

from utils import (
    DEFAULT_PDF_DIR,
    create_qa_chain,
    interactive_qa,
    load_or_create_index,
    pretty_print_sources,
)


def run_cli(pdf_dir: Path, index_dir: Path, question: str | None, rebuild: bool) -> None:
    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"PDF directory '{pdf_dir}' does not exist. Put your PDFs under that path or pass --pdf-dir."
        )

    vector_store = load_or_create_index(pdf_dir, index_dir=index_dir, force_rebuild=rebuild)
    qa_chain = create_qa_chain(vector_store)

    if question:
        result = qa_chain({"query": question})
        print("\nAssistant:")
        print(result["result"])
        pretty_print_sources(result.get("source_documents", []))
        return

    interactive_qa(qa_chain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions against your PDFs with a simple RAG pipeline.")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR, help="Directory containing PDF files.")
    parser.add_argument("--index-dir", type=Path, default="faiss_index", help="Directory to store the FAISS index.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the index even if it exists.")
    parser.add_argument("--question", type=str, help="Optional single question (skips interactive mode).")
    args = parser.parse_args()

    run_cli(pdf_dir=args.pdf_dir, index_dir=args.index_dir, question=args.question, rebuild=args.rebuild)

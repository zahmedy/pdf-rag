import os
import warnings
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings(action="ignore")
load_dotenv()

# Default locations can be overridden via function args.
INDEX_DIR = Path("faiss_index")
DEFAULT_PDF_DIR = Path("data") / "docs"


def ensure_api_key() -> None:
    """Raise an error early if the OpenAI API key is missing."""
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is missing. Add it to your environment or .env file."
        )


def load_pdfs(pdf_dir: Path) -> List:
    """Load all PDF files from a directory into LangChain document objects."""
    print(f"Loading PDFs from: {pdf_dir}")
    loader = PyPDFDirectoryLoader(str(pdf_dir))
    documents = loader.load()
    print(f"Loaded {len(documents)} documents/pages")
    return documents


def split_documents(documents: Iterable, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split documents into small overlapping chunks for better retrieval."""
    print(f"Splitting into chunks: size={chunk_size}, overlap={chunk_overlap}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def build_faiss_index(chunks, index_dir: Path = INDEX_DIR) -> FAISS:
    """Build a FAISS index from document chunks using OpenAI embeddings."""
    ensure_api_key()

    print("Creating OpenAI embeddings...")
    embedding_model = OpenAIEmbeddings()

    print("Building FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embedding_model)

    print(f"Saving FAISS index to: {index_dir}")
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))
    return vector_store


def load_or_create_index(pdf_dir: Path, index_dir: Path = INDEX_DIR, force_rebuild: bool = False) -> FAISS:
    """Load an existing FAISS index, or build one from PDF files if missing."""
    if index_dir.exists() and not force_rebuild:
        print(f"Loading existing FAISS index from {index_dir}...")
        ensure_api_key()
        embedding_model = OpenAIEmbeddings()
        return FAISS.load_local(
            str(index_dir),
            embedding_model,
            allow_dangerous_deserialization=True,
        )

    docs = load_pdfs(pdf_dir)
    chunks = split_documents(docs)
    return build_faiss_index(chunks, index_dir=index_dir)


def create_qa_chain(vector_store: FAISS):
    """
    Create a RetrievalQA chain that:
      - retrieves similar chunks from FAISS
      - passes them + question to ChatOpenAI
    """
    ensure_api_key()
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )
    return qa_chain


def pretty_print_sources(sources: List) -> None:
    """Display which documents were used to answer the query."""
    if not sources:
        print("No source documents returned.")
        return

    print("\nSources:")
    for i, doc in enumerate(sources, start=1):
        meta = doc.metadata
        page = meta.get("page", "N/A")
        source = meta.get("source", meta.get("file_path", "N/A"))
        print(f" [{i}] {source} (page {page})")


def interactive_qa(qa_chain: RetrievalQA) -> None:
    """CLI loop for asking questions against the built index."""
    print("\nPDF Knowledge Assistant (RAG)")
    print("Type 'exit', 'quit', or Ctrl+C to stop.\n")

    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            if not query:
                continue

            result = qa_chain({"query": query})
            answer = result["result"]
            sources = result.get("source_documents", [])

            print("\nAssistant:")
            print(answer)
            pretty_print_sources(sources)
            print("-" * 60)

        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break

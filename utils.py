import os
from dotenv import load_dotenv 

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()
INDEX_DIR = "faiss_index"

def load_pdfs(pdf_dir: str):
    """
    Load all PDF files from a directory into LangChain DOcumet objects.
    """
    print(f"Loading PDFs from: {pdf_dir}")
    loader = PyPDFDirectoryLoader(pdf_dir)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents/pages")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into small overlapping chunks for better retrieval.
    """
    print(f"Splitting into chunks: size={chunk_size}, overlap={chunk_overlap}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def build_faiss_index(chunks, INDEX_DIR=INDEX_DIR):
    """
    Build a FAISS index from documents chunks using OpenAI embeddings. 
    """
    print("Creating OpenAI embeddings...")
    embedding_model = OpenAIEmbeddings()  # uses OPENAI_API_KEY

    print("Building FAISS vector store...")
    vectore_store = FAISS.from_documents(chunks, embedding_model)

    # Sabe locally so we don't rebuild every run
    print(f"Saving FAISS index to: {INDEX_DIR}")

    return vectore_store

def load_or_create_index(pdf_dir: str, INDEX_DIR=INDEX_DIR):
    """
    Try to load existing FAISS index; if not found, build it from PDF.
    """
    if os.path.exists(INDEX_DIR):
        print(f"Loading existing FAISS index from {INDEX_DIR}...")
        embedding_model = OpenAIEmbeddings()
        vector_store = FAISS.load_local(
            INDEX_DIR,
            embedding_model,
            allow_dangerous_deserialization=True,
        )

        return vector_store
    
    # Build index from scratch 
    docs = load_pdfs(pdf_dir)
    chunks = split_documents(docs)
    vector_store = build_faiss_index(chunks)
    return vector_store

def create_qa_chain(vectore_store: FAISS):
    """
    Create a RetrievalQA chain that:
      - retrieves similar chunks from FAISS
      - passes them + question to ChatOpenAI
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0         # more deterministic
    )

    retriever = vectore_store.as_retriever(
        search_type="similarity",   # basic similarity search
        search_kwargs={"k": 4},     # number of chunks to retrieve
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,   # so we can inspect which chunks were used
        chain_type="stuff"              # simplest: stuff all chunks into context
    )

    return qa_chain

def interactive_qa(qa_chain):
    print("\nPDF Knowledge Assistant (RAG)")
    print("Type 'exit', 'quit', or Ctrl+C to stop.\n")

    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in {"exist", "quit"}:
                print("Goodbye...")
                break
        
            if not query:
                continue

            result = qa_chain({"query": query})
            answer = result["result"]
            sources = result.get("source_documents", [])

            print("\nAssistant:")
            print(answer)
            print("\nSource:")
            for i, doc in enumerate(sources, start=1):
                meta = doc.metadata
                page = meta.get("page", "N/A")
                source = meta.get("Source", "N/A")
                print(f" [{i}] {source} (page {page})")

            print("-" * 60)

        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break

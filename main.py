
from utils import *

if __name__ == "__main__":
    pdf_dir = os.path.join("data", "docs")
    vector_store = load_or_create_index(pdf_dir)
    qa_chain = create_qa_chain(vector_store)
    interactive_qa(qa_chain)
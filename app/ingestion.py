import os
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------
# Configuration
# -----------------------------

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# Load PDF documents
# -----------------------------

def load_documents():

    documents = []

    pdf_files = glob.glob(
        os.path.join(DATA_DIR, "*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found inside '{DATA_DIR}'"
        )

    print("\nPDF files found:")

    for pdf_file in pdf_files:

        print(f"  - {pdf_file}")

        loader = PyPDFLoader(pdf_file)

        docs = loader.load()

        documents.extend(docs)

    print(f"\nTotal pages loaded: {len(documents)}")

    return documents


# -----------------------------
# Split documents
# -----------------------------

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = text_splitter.split_documents(
        documents
    )

    print(f"Total chunks created: {len(chunks)}")

    return chunks


# -----------------------------
# Create embeddings
# -----------------------------

def create_embeddings():

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    return embeddings


# -----------------------------
# Create vector database
# -----------------------------

def create_vectorstore():

    print("\nStarting document ingestion...")

    # Load documents
    documents = load_documents()

    # Split documents
    chunks = split_documents(documents)

    # Embeddings
    embeddings = create_embeddings()

    print("\nCreating FAISS vector database...")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Create directory
    os.makedirs(
        VECTORSTORE_DIR,
        exist_ok=True
    )

    # Save vector database
    vectorstore.save_local(
        VECTORSTORE_DIR
    )

    print("\nVector database created successfully.")

    print(
        f"Saved to: {VECTORSTORE_DIR}"
    )

    return vectorstore


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    create_vectorstore()
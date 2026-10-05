import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------
# Configuration
# -----------------------------

VECTORSTORE_DIR = "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# Create embedding model
# -----------------------------

def get_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return embeddings


# -----------------------------
# Load vector database
# -----------------------------

def load_vectorstore():

    if not os.path.exists(
        VECTORSTORE_DIR
    ):

        raise FileNotFoundError(
            "Vector database not found.\n"
            "Run ingestion.py first."
        )

    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


# -----------------------------
# Create retriever
# -----------------------------

def get_retriever():

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    return retriever


# -----------------------------
# Retrieve documents
# -----------------------------

def retrieve_documents(
    query,
    retriever=None
):

    if retriever is None:

        retriever = get_retriever()

    documents = retriever.invoke(
        query
    )

    return documents


# -----------------------------
# Test retrieval
# -----------------------------

if __name__ == "__main__":

    retriever = get_retriever()

    query = input(
        "\nEnter your question: "
    )

    documents = retrieve_documents(
        query,
        retriever
    )

    print("\nRetrieved Documents:\n")

    for index, doc in enumerate(
        documents,
        start=1
    ):

        print(
            f"\n--- Result {index} ---"
        )

        print(
            doc.page_content
        )

        print(
            "Metadata:",
            doc.metadata
        )
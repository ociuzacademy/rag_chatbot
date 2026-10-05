from retrieval import (
    get_retriever,
    retrieve_documents
)

from llm import generate_answer


# -----------------------------
# Create context
# -----------------------------

def create_context(documents):

    context_parts = []

    for document in documents:

        context_parts.append(
            document.page_content
        )

    context = "\n\n".join(
        context_parts
    )

    return context


# -----------------------------
# Create source information
# -----------------------------

def get_sources(documents):

    sources = []

    for document in documents:

        metadata = document.metadata

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page",
            None
        )

        if page is not None:

            page = page + 1

            source_info = (
                f"{source} - Page {page}"
            )

        else:

            source_info = source

        if source_info not in sources:

            sources.append(
                source_info
            )

    return sources


# -----------------------------
# Ask question
# -----------------------------

def ask_question(
    question,
    retriever
):

    # Retrieve relevant documents
    documents = retrieve_documents(
        question,
        retriever
    )

    # Create context
    context = create_context(
        documents
    )

    # Generate answer
    answer = generate_answer(
        question,
        context
    )

    # Get sources
    sources = get_sources(
        documents
    )

    return answer, sources


# -----------------------------
# Main chatbot
# -----------------------------

def main():

    print("=" * 60)
    print("        RAG DOCUMENT CHATBOT")
    print("=" * 60)

    print(
        "\nLoading vector database..."
    )

    try:

        retriever = get_retriever()

    except Exception as e:

        print(
            "\nError loading vector database:"
        )

        print(e)

        print(
            "\nRun ingestion.py first."
        )

        return

    print(
        "Vector database loaded successfully."
    )

    print(
        "\nType 'exit' to close the chatbot."
    )

    while True:

        question = input(
            "\nYou: "
        ).strip()

        if not question:

            continue

        if question.lower() == "exit":

            print(
                "\nGoodbye! 👋"
            )

            break

        try:

            answer, sources = ask_question(
                question,
                retriever
            )

            print(
                "\nAssistant:"
            )

            print(answer)

            print(
                "\nSources:"
            )

            for source in sources:

                print(
                    f"  - {source}"
                )

        except Exception as e:

            print(
                "\nError:"
            )

            print(e)


# -----------------------------
# Run application
# -----------------------------

if __name__ == "__main__":

    main()
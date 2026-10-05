import os

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI
)

from langchain_core.prompts import (
    ChatPromptTemplate
)


# -----------------------------
# Load environment variables
# -----------------------------

load_dotenv()


# -----------------------------
# Check API key
# -----------------------------

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)

if not GOOGLE_API_KEY:

    raise ValueError(
        "GOOGLE_API_KEY not found.\n"
        "Add it to your .env file."
    )


# -----------------------------
# Create LLM
# -----------------------------

def get_llm():

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    return llm


# -----------------------------
# RAG Prompt
# -----------------------------

def get_prompt():

    prompt = ChatPromptTemplate.from_template(
        """
You are an intelligent document assistant.

Your task is to answer the user's question
using ONLY the information provided in the
context below.

Do not use outside knowledge.

If the answer cannot be found in the context,
clearly say:

"I couldn't find this information in the
provided documents."

Do not make up information.

Keep the answer clear, accurate and concise.

-------------------------
CONTEXT
-------------------------

{context}

-------------------------
QUESTION
-------------------------

{question}

-------------------------
ANSWER
-------------------------
"""
    )

    return prompt


# -----------------------------
# Generate answer
# -----------------------------

def generate_answer(
    question,
    context
):

    llm = get_llm()

    prompt = get_prompt()

    formatted_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    response = llm.invoke(
        formatted_prompt
    )

    return response.content
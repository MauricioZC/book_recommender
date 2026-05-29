from dotenv import load_dotenv
import numpy as np
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src import (
    UserPrompt,
    enhance_prompt,
    encode_prompt,
    similarity_score,
    get_table,
)

load_dotenv()

MODEL = "gpt-4.1-nano"

SYSTEM_PROMPT = """You are a knowledgeable, friendly book recommender.

You have one tool: `search_books`. Call it when the user wants new
recommendations or describes a book they're looking for. Do NOT call it
for follow-up questions about books already shown in this conversation
(e.g. "tell me more about the second one", "which is shortest?",
"compare the first two") — answer those from prior context.

When you do recommend books, only recommend ones returned by `search_books`.
Never invent titles, authors, or details.
"""


def _format_books_for_llm(top_books: pd.DataFrame) -> str:
    """Turn the top_books DataFrame into a string the LLM can read."""
    if top_books is None or len(top_books) == 0:
        return "No matching books found."
    lines = []
    for i, r in enumerate(top_books.itertuples(index=False), 1):
        lines.append(f"{i}. {r.title} by {r.authors} — {r.categories}")
    return "\n".join(lines)


def make_search_tool(books_df: pd.DataFrame, embeddings_matrix: np.ndarray):
    """Build the search_books tool, closing over the loaded data."""

    @tool
    def search_books(query: str) -> str:
        """Search the book database for recommendations matching the user's query.

        Use this when the user describes what they want to read (genre, theme,
        mood, comparable books). Returns up to 5 matching books with title,
        author, and category.

        Args:
            query: The user's description of what they want to read.
        """
        user_prompt = UserPrompt(query=query)
        enhanced = enhance_prompt(user_prompt)
        query_embedding = encode_prompt(enhanced)
        scores = similarity_score(query_embedding, embeddings_matrix)
        top_books = get_table(books_df, scores, top_n=5)
        return _format_books_for_llm(top_books)

    return search_books


class BookChat:
    """Wraps a LangChain tool-calling agent with conversation history."""

    def __init__(self, books_df: pd.DataFrame, embeddings_matrix: np.ndarray):
        llm = ChatOpenAI(temperature=0, model=MODEL)
        tools = [make_search_tool(books_df, embeddings_matrix)]

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        self.executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        self.history: list = []

    def message(self, user_input: str) -> str:
        """Send a user message and return the assistant's reply."""
        result = self.executor.invoke({
            "input": user_input,
            "chat_history": self.history,
        })
        reply = result["output"]
        self.history.append(HumanMessage(content=user_input))
        self.history.append(AIMessage(content=reply))
        return reply


# --- Dev-only: launch a Gradio chat to demo this module ---

def _demo():
    import gradio as gr

    books_df = pd.read_csv("books.csv").fillna("")
    embeddings_matrix = np.load("embeddings.npy")
    chat = BookChat(books_df, embeddings_matrix)

    def respond(question, history):
        return chat.message(question)

    gr.ChatInterface(
        respond,
        title="📚 Book Recommender",
        description="Tell me what you're in the mood to read.",
        examples=[
            "A dark fantasy about power and sacrifice",
            "Books like Harry Potter",
            "Short literary novels about grief",
        ],
    ).launch()


if __name__ == "__main__":
    _demo()
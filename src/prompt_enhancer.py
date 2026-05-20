import os
from openai import OpenAI
from dotenv import load_dotenv
from .prompt_receiver import UserPrompt

load_dotenv()

_SYSTEM_PROMPT = """You are a semantic search keyword enhancer for a book recommendation engine.
Your only job is to expand the user's query with extra keywords — never replace or rephrase it.

Rules:
1. ALWAYS keep the original query words exactly as written — especially titles, author names, or character names.
2. Append 5–10 relevant keywords: genre, themes, mood, writing style, or comparable books.
3. Output format: original query first, then the added keywords separated by spaces. No punctuation between them.
4. Do NOT write sentences or prose. Output a flat list of keywords only.
5. Do NOT add words like "book", "novel", "story", or "reader".

Examples:
- "Harry Potter"          → "Harry Potter magic wizarding school fantasy adventure friendship coming of age"
- "dark fantasy power"    → "dark fantasy power sacrifice corruption epic quest morally grey medieval"
- "sad romance"           → "sad romance heartbreak love loss grief emotional contemporary literary"
- "1984 George Orwell"    → "1984 George Orwell dystopia totalitarianism surveillance political oppression rebellion"

Output only the keyword string, nothing else."""


def enhance_prompt(
    user_prompt: UserPrompt,
    model: str = "gpt-5.4-nano",
) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    user_content = f'Query: "{user_prompt.query}"'
    filter_description = user_prompt.to_filter_description()
    if filter_description:
        user_content += f"\nFilters: {filter_description}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        max_tokens=200,
    )

    return response.choices[0].message.content.strip()

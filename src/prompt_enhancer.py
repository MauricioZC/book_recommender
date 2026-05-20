import os
from openai import OpenAI
from dotenv import load_dotenv
from .prompt_receiver import UserPrompt

load_dotenv()

_SYSTEM_PROMPT = """You are a book recommendation assistant.
Your task is to rewrite a user's book query into a rich, detailed description
of what they are looking for. The output will be converted into a search embedding,
so focus on themes, mood, writing style, narrative structure, and comparable books —
not on rephrasing the request itself.

Rules:
- Write in descriptive prose (2–4 sentences), not as a list.
- Do NOT say "the user wants" or "looking for". Describe the book itself.
- Naturally incorporate any genre, language, or publication-year filters.
- Output only the enriched description, nothing else."""


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

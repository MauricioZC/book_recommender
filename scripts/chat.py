import json
from openai import OpenAI
from dotenv import load_dotenv
import inspect
import pandas as pd
from pathlib import Path

SYSTEM_PROMPT = """
    You are a book recommender assistant. You help users find books by calling the search_books tool.

    Rules:
    - Always call search_books before recommending any book.
    - If search_books returns no results (None, empty, or no matches), do NOT recommend any books from your own knowledge. Tell the user no matches were found and suggest they broaden their criteria (e.g., wider year range, different genre).
    - Only recommend books that appear in the tool's results. Never invent titles, authors, or details.
"""
DB_PATH = Path(__file__).parent.parent / "books.csv"
load_dotenv()


def get_genres(df):
    return df["categories"].str.lower().dropna().unique().tolist()


def filter_books():
    ... # TODO


def similarity_search():
    ... # TODO


genres = get_genres(pd.read_csv(DB_PATH))


# This should go in a separate file
tools = [{
    "type": "function",
    "function": {
        "name": "search_books",
        "description": "Query the book database to narrow down the search space for a similarity search in a book recommender application",
        "parameters": {
            "type": "object",
            "properties": {
                "genre": {
                    "type": "string",
                    "description": "The genre of the book",
                    "enum": genres
                },
                "year_min": {
                    "type": "integer",
                    "description": "Earliest publication year"
                },
                "year_max": {
                    "type": "integer",
                    "description": "Latest publication year"
                },
                "query": {
                    "type": "string",
                    "description": "The user's raw search query, passed directly to the embedding model"
                }
            },
            "required": ["query"]
        }
    }
}]


def access_db(path):
    df = pd.read_csv(DB_PATH)
    return df


def search_books(genre=None, year_min=None, year_max=None, query=None, path=None):
    print("TOOL CALLED: search_books()")
    df = access_db(DB_PATH)

    if genre is not None:
        df = df[df["categories"].str.lower() == genre.lower()]
    if year_min is not None:
        df = df[df["published_year"] >= float(year_min)]
    if year_max is not None:
        df = df[df["published_year"] <= float(year_max)]

    # Vectorize the user query
    ...  # TODO

    return df.head(5).to_json(orient="records")


class Chat:
    def __init__(self, model, system_prompt=SYSTEM_PROMPT, tools=None, tool_functions=None, tool_choice="auto"):
        self.model = model
        self.client = OpenAI()
        self.tools = tools
        self.tool_functions = tool_functions or {}
        self.tool_choice = tool_choice
        self.available_functions = {
                "search_books": search_books,
            } 
        self.system_prompt = system_prompt
        self.messages = []
        self.history = [] # TODO, implement
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def message(self, prompt):
        self.messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}] # This might be buggy, we dont want to overwrite the history every user prompt

        # Test print: Checks message history
        print("\nInitial Message: ", self.messages)

        response = self.client.chat.completions.create(
            model=self.model, 
            messages=self.messages,
            tools=self.tools,
            tool_choice=self.tool_choice
        )

        response_message = response.choices[0].message
        self.tool_calls = response_message.tool_calls

        if self.tool_calls:
            # Test print: Checks tool calls
            print(self.tool_calls)
            # TODO
            self.call_tools(response_message)
        else:
            return response_message.content

    def call_tools(self, response_message):        
        self.messages.append(response_message)

        # Call the function and add the response
        for tool_call in self.tool_calls:
            function_name = tool_call.function.name
            function_to_call = self.available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            
            # Get the function signature and call the function with given arguments
            sig = inspect.signature(function_to_call)
            call_args = {
                k: function_args.get(k, v.default)
                for k, v in sig.parameters.items()
                if k in function_args or v.default is not inspect.Parameter.empty
            }
            print(f"\nCalling {function_to_call} with arguments {call_args}")
            
            function_response = str(function_to_call(**call_args))
            
            print("\nFunction Response: ", function_response)

            # Put output into a tool message
            tool_message = {
                    "tool_call_id": tool_call.id, # Needed for Parallel Tool Calling
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            print("\nAppending Message: ", tool_message)
            
            # Extend conversation with function response
            self.messages.append(tool_message)  

        # Get a new response from the model where it can see the entire conversation including the function call outputs
        second_response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
        )  

        print("\nLLM Response: ", second_response)

        print("\n---Formatted LLM Response---")
        print("\n",second_response.choices[0].message.content)
        
        return
    

# Test "Chat" class
agent = Chat("gpt-4o-mini", tools=tools)
prompt = input("Chat: ").lower()

response = agent.message(prompt)
print(response)





from inspect_ai.model._chat_message import ChatMessageSystem, ChatMessageUser, ChatMessage
from inspect_ai.model._model import get_model
import asyncio
import sys

model_name = "openai/gpt-4o-mini" # https://inspect.ai-safety-institute.org.uk/
system_message = "You are a kind robot"

async def get_response(question: str) -> str:
    messages = [
            ChatMessageSystem(content=system_message),
            ChatMessageUser(content=question),
    ]
    model = get_model(model_name)
    response = await model.generate(input=messages)
    print(response.message.text)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python chat.py \"your question here\"")
        sys.exit(1)
    
    question = sys.argv[1]
    asyncio.run(get_response(question))
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from ai_agent import get_response_from_ai_agent
from dotenv import load_dotenv
import uvicorn

# Load environment variables (if using a .env file)
load_dotenv()

# Allowed AI model names
ALLOWED_MODEL_NAMES = [
    "deepseek-r1-distill-llama-70b",
    "mixtral-8x7b-32768",
    "llama-3.3-70b-versatile",
    "gpt-4o-mini"
]

# FastAPI app instance
app = FastAPI(title="LangGraph AI Agent")

# Request schema validation
class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API Endpoint to interact with the chatbot using LangGraph.
    It dynamically selects the model specified in the request.
    """
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model."}

    # Extract parameters from the request
    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider

    # Get AI response
    response = get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider)
    return response

# Run the server when executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9999, reload=True)

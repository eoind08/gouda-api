from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import load_model, generate_response, model_cache
import os

app = FastAPI()

class MessageRequest(BaseModel):
    model_name: str
    message: str

@app.post("/chat")
def chat(request: MessageRequest):
    model_name = request.model_name
    user_message = request.message
    model_path = f"models/{model_name}/model.pt"

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    if model_name not in model_cache:
        model_cache[model_name] = load_model(model_name)

    try:
        response = generate_response(model_cache[model_name], user_message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

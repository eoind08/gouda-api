from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import load_model, generate_response, model_cache
import os
from huggingface_hub import hf_hub_download

app = FastAPI()

class MessageRequest(BaseModel):
    model_name: str
    message: str

# Map model_name from request to Hugging Face repo ID
HF_REPO_MAP = {
    "gouda0.0.1": "Erbium08/gouda0.0.1",
    # Add more here if needed:
    # "cheddar1.0": "Erbium08/cheddar1.0"
}

def download_model(model_name):
    if model_name not in HF_REPO_MAP:
        raise ValueError(f"Unknown model: {model_name}")

    repo_id = HF_REPO_MAP[model_name]

    # Create local cache directory
    model_dir = f"models/{model_name}"
    os.makedirs(model_dir, exist_ok=True)

    # Download model.pt from Hugging Face Hub
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.pt",
        cache_dir=model_dir,
        local_dir=model_dir,
        force_filename="model.pt"
    )

    return model_path

@app.post("/chat")
def chat(request: MessageRequest):
    model_name = request.model_name
    user_message = request.message
    model_path = download_model(model_name)

    if model_name not in model_cache:
        model_cache[model_name] = load_model(model_name)

    try:
        response = generate_response(model_cache[model_name], user_message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

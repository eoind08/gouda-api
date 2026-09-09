from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_loader import (
    get_model,
    generate_response,
    is_model_available,
    list_models,
)


app = FastAPI(
    title="Gouda API",
    version="1.0.0",
)


# ============================================================
# Request / Response models
# ============================================================

class MessageRequest(BaseModel):
    model_name: str
    message: str
    max_tokens: int = 100


# ============================================================
# Health check
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "available_models": list_models(),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
    }


# ============================================================
# Chat
# ============================================================

@app.post("/chat")
def chat(request: MessageRequest):

    model_name = request.model_name.strip()

    if not model_name:
        raise HTTPException(
            status_code=400,
            detail="model_name cannot be empty",
        )

    if not is_model_available(model_name):
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Unknown model",
                "model": model_name,
                "available_models": list_models(),
            },
        )

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="message cannot be empty",
        )

    if not 1 <= request.max_tokens <= 256:
        raise HTTPException(
            status_code=400,
            detail="max_tokens must be between 1 and 256",
        )

    try:

        model = get_model(
            model_name
        )

        response = generate_response(
            model=model,
            message=request.message,
            max_tokens=request.max_tokens,
        )

        return {
            "response": response,
            "model": model_name,
        }

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        print(
            f"[ERROR] Generation failed "
            f"for {model_name}: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Model generation failed.",
        )
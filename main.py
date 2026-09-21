from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from model_loader import (
    generate_response,
    generate_response_stream,
    is_model_available,
    list_models,
)


app = FastAPI(
    title="Gouda API",
    version="1.3.0",
)


class MessageRequest(BaseModel):
    model_name: str = "Gruyere-1.1"
    message: str

    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 40

    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    seed: int | None = None
    stop: list[str] | None = None


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

    if not 0 <= request.temperature <= 2:
        raise HTTPException(
            status_code=400,
            detail="temperature must be between 0 and 2",
        )

    if not 0 < request.top_p <= 1:
        raise HTTPException(
            status_code=400,
            detail="top_p must be greater than 0 and at most 1",
        )

    if request.top_k < 0:
        raise HTTPException(
            status_code=400,
            detail="top_k cannot be negative",
        )

    if request.repetition_penalty <= 0:
        raise HTTPException(
            status_code=400,
            detail="repetition_penalty must be greater than 0",
        )

    try:
        result = generate_response(
            model_name=model_name,
            message=request.message,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            seed=request.seed,
            stop=request.stop,
        )

        return {
            "response": result["text"],
            "model": model_name,
            "usage": result["usage"],
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


@app.post("/chat/stream")
def chat_stream(request: MessageRequest):

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

    if not 0 <= request.temperature <= 2:
        raise HTTPException(
            status_code=400,
            detail="temperature must be between 0 and 2",
        )

    if not 0 < request.top_p <= 1:
        raise HTTPException(
            status_code=400,
            detail="top_p must be greater than 0 and at most 1",
        )

    if request.top_k < 0:
        raise HTTPException(
            status_code=400,
            detail="top_k cannot be negative",
        )

    if request.repetition_penalty <= 0:
        raise HTTPException(
            status_code=400,
            detail="repetition_penalty must be greater than 0",
        )

    try:
        return StreamingResponse(
            generate_response_stream(
                model_name=model_name,
                message=request.message,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                seed=request.seed,
                stop=request.stop,
            ),
            media_type="text/plain",
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        print(
            f"[ERROR] Streaming generation failed "
            f"for {model_name}: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Model generation failed.",
        )
import os
import json
import uuid
import asyncio # <-- Added import
import time # Ensure time is imported for timestamps
from typing import List, Optional, Literal, Dict, Any, AsyncGenerator

import httpx
import dotenv
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Load environment variables from .env file
dotenv.load_dotenv()

# --- Configuration ---
FLOWITH_AUTH_TOKEN = os.getenv("FLOWITH_AUTH_TOKEN")
FLOWITH_API_URL = "https://edge.flo.ing/completion?mode=general"
MODELS_FILE_PATH = "models.json"
API_KEY = os.getenv("API_KEY", "123456") # Read API Key

# --- Security Scheme ---
security = HTTPBearer()

if not FLOWITH_AUTH_TOKEN:
    # In a real app, you might want to log this and exit,
    # but for simplicity, we'll raise an error if accessed.
    print("Warning: FLOWITH_AUTH_TOKEN environment variable not set.")
    # raise ValueError("FLOWITH_AUTH_TOKEN environment variable is required.") # Or handle differently

# --- Load Model Mappings ---
try:
    with open(MODELS_FILE_PATH, 'r') as f:
        model_mappings = json.load(f)
except FileNotFoundError:
    print(f"Error: Models file not found at {MODELS_FILE_PATH}")
    model_mappings = {} # Or raise an error / exit
except json.JSONDecodeError:
    print(f"Error: Invalid JSON in models file: {MODELS_FILE_PATH}")
    model_mappings = {} # Or raise an error / exit

# --- Pydantic Models ---
class OpenAIMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class OpenAIRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    stream: Optional[bool] = False
    # Add other potential OpenAI fields if needed, e.g., temperature, max_tokens
    # temperature: Optional[float] = None
    # max_tokens: Optional[int] = None

class FlowithMessage(BaseModel):
    role: Literal["user", "assistant"] # Flowith uses 'user' and 'assistant'
    content: str

class FlowithRequest(BaseModel):
    model: str
    messages: List[FlowithMessage]
    stream: bool
    nodeId: str # UUID for Flowith


# --- OpenAI Models Endpoint Models ---
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    # owned_by: str = "user" # Optional: Add other fields if needed

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

# --- FastAPI App ---
app = FastAPI(
    title="OpenAI to Flowith Proxy",
    description="Translates OpenAI-compatible chat completion requests to Flowith's format.",
)

# --- Helper for Streaming ---
async def stream_flowith_response(flowith_stream: httpx.Response) -> AsyncGenerator[str, None]:
    """Asynchronously streams the response from Flowith."""
    async for chunk in flowith_stream.aiter_bytes():
        # Assuming Flowith streams data in a format compatible with OpenAI's SSE
        # If Flowith uses a different streaming format, this needs adjustment.
        yield chunk.decode('utf-8') # Decode bytes to string

# --- Security Dependency ---
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the provided API key against the environment variable."""
    # Use constant time comparison to prevent timing attacks
    # This requires Python 3.3+
    import hmac
    is_valid = hmac.compare_digest(credentials.credentials, API_KEY)

    if not credentials or credentials.scheme != "Bearer" or not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials # Return the key or True if successful

# --- API Endpoint ---
@app.post("/v1/chat/completions")
async def chat_completions(
    request: OpenAIRequest,
    http_request: Request,
    api_key: str = Depends(verify_api_key) # Add API Key dependency
):
    """
    Accepts OpenAI-like chat completion requests and forwards them to Flowith.
    """
    if not FLOWITH_AUTH_TOKEN:
         raise HTTPException(status_code=500, detail="Server configuration error: Flowith auth token not set.")

    # 1. Map the model
    flowith_model_name = model_mappings.get(request.model)
    if not flowith_model_name:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not found in mappings. Available: {list(model_mappings.keys())}"
        )

    # 2. Generate nodeId
    node_id = str(uuid.uuid4())

    # 3. Process messages (Handle system prompt for specific models)
    processed_messages: List[FlowithMessage] = []
    # Check if the *target* Flowith model indicates Claude or Gemini
    # Adjust this logic if the check should be based on the *source* OpenAI model name
    is_claude_or_gemini = "claude" in flowith_model_name.lower() or "gemini" in flowith_model_name.lower()

    for msg in request.messages:
        original_role = msg.role.lower() # Get original role
        new_role = original_role # Start with original role

        # Check for Claude/Gemini system message conversion based on *requested* model
        is_claude_or_gemini_requested = "claude" in request.model.lower() or "gemini" in request.model.lower()
        if is_claude_or_gemini_requested and original_role == "system":
            new_role = "user"
        # Check for non-standard roles (applies AFTER potential system->user conversion for C/G)
        elif original_role not in {"user", "assistant", "system"}:
             new_role = "user"
             print(f"Warning: Converting non-standard role '{original_role}' to 'user' for model {request.model}")

        # Append message with the determined new_role, but only if it's valid for Flowith
        # Flowith only accepts 'user' and 'assistant'
        if new_role in ["user", "assistant"]:
             processed_messages.append(FlowithMessage(role=new_role, content=msg.content))
        # else: # Log or skip roles that are still 'system' after processing
        #    print(f"Skipping message with final role '{new_role}' as it's not 'user' or 'assistant'. Original role: '{original_role}'")

    # 4. Construct Flowith Request Payload
    flowith_payload = FlowithRequest(
        model=flowith_model_name,
        messages=processed_messages,
        stream=True, # Always stream from Flowith
        nodeId=node_id,
    )

    # 5. Prepare Headers for Flowith Request
    # Headers exactly matching the curl -H flags provided
    headers = {
        'accept': 'text/event-stream', # Changed for streaming
        'accept-language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,zh-TW;q=0.6,ja;q=0.5',
        'authorization': FLOWITH_AUTH_TOKEN, # Send only the token, no "Bearer " prefix
        'content-type': 'application/json',
        'responsetype': 'stream', # Keep this header
        'origin': 'https://flowith.net',
        'priority': 'u=1, i',
        'referer': 'https://flowith.net/',
        'sec-ch-ua': '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'cross-site',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36'
    }

    # 6. Make Asynchronous Request to Flowith
    # Need time for simulated streaming chunks
    import time
    # Need JSONResponse
    from fastapi.responses import JSONResponse

    # Need JSONResponse
    from fastapi.responses import JSONResponse

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # Serialize payload manually for the stream request
            payload_json = flowith_payload.dict()

            # Use client.stream for real streaming
            async with client.stream("POST", FLOWITH_API_URL, headers=headers, json=payload_json, timeout=300.0) as response:
                # Check status *after* starting stream read or getting headers
                # It's often better to check after reading the first chunk or headers
                # For simplicity here, we might check early, but be aware Flowith might send errors mid-stream
                # Let's check status right away, but handle potential errors during iteration too.
                try:
                    response.raise_for_status() # Check initial status
                except httpx.HTTPStatusError as status_exc:
                     # Attempt to read body for more details if possible
                    try:
                        error_body = await status_exc.response.aread()
                        detail_msg = f"Flowith API Error ({status_exc.response.status_code}): {error_body.decode('utf-8', errors='replace')}"
                    except Exception:
                        detail_msg = f"Flowith API Error ({status_exc.response.status_code})"
                    raise HTTPException(status_code=status_exc.response.status_code, detail=detail_msg) from status_exc


                # 7. Handle response based on *client's* request.stream preference
                if request.stream:
                    # Client wants streaming: Forward Flowith's stream directly
                    async def stream_forwarder():
                        try:
                            async for chunk in response.aiter_bytes():
                                # Assuming Flowith sends SSE-compatible chunks (or raw data we want to forward)
                                # If Flowith sends *only* text content per chunk, we might need to format it:
                                # yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk.decode()}}]})}\n\n"
                                # For now, let's assume Flowith sends pre-formatted SSE or raw bytes are okay.
                                yield chunk
                            # Optionally yield a final [DONE] message if Flowith doesn't
                            # yield b"data: [DONE]\n\n"
                        except httpx.RequestError as stream_exc:
                            print(f"Error during Flowith stream read: {stream_exc}")
                            # Decide how to signal this error to the client.
                            # Option 1: Raise an exception (FastAPI might handle it)
                            # raise HTTPException(status_code=503, detail=f"Stream read error: {stream_exc}")
                            # Option 2: Yield an error message within the stream (if client supports it)
                            error_content = f"Stream read error: {type(stream_exc).__name__}"
                            error_chunk = {"id": f"chatcmpl-streamerror-{uuid.uuid4()}", "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"delta": {"content": error_content }, "index": 0, "finish_reason": "error"}]}
                            yield f"data: {json.dumps(error_chunk)}\n\n".encode('utf-8')
                            yield b"data: [DONE]\n\n" # Still send DONE after error chunk
                        except Exception as e:
                            print(f"Unexpected error during stream forward: {e}")
                            error_content = f"Unexpected stream error: {type(e).__name__}"
                            error_chunk = {"id": f"chatcmpl-streamerror-{uuid.uuid4()}", "object": "chat.completion.chunk", "created": int(time.time()), "model": request.model, "choices": [{"delta": {"content": error_content }, "index": 0, "finish_reason": "error"}]}
                            yield f"data: {json.dumps(error_chunk)}\n\n".encode('utf-8')
                            yield b"data: [DONE]\n\n" # Still send DONE after error chunk


                    return StreamingResponse(stream_forwarder(), media_type="text/event-stream")

                else:
                    # Client wants non-streaming: Accumulate chunks
                    chunks = []
                    try:
                        async for chunk in response.aiter_bytes():
                            chunks.append(chunk)
                    except httpx.RequestError as stream_exc:
                         print(f"Error during Flowith stream read (non-streaming mode): {stream_exc}")
                         raise HTTPException(status_code=503, detail=f"Stream read error: {stream_exc}")
                    except Exception as e:
                         print(f"Unexpected error during stream accumulation: {e}")
                         raise HTTPException(status_code=500, detail=f"Unexpected stream error: {e}")


                    full_response_bytes = b"".join(chunks)
                    flowith_text = full_response_bytes.decode('utf-8', errors='replace')
                    print(f"Accumulated response preview: {flowith_text[:500]}")

                    # Construct OpenAI-compatible JSON
                    completion_id = f"chatcmpl-{uuid.uuid4()}"
                    created_timestamp = int(time.time())
                    response_payload = {
                        "id": completion_id,
                        "object": "chat.completion",
                        "created": created_timestamp,
                        "model": request.model, # Use the model from the original request
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": flowith_text
                            },
                            "finish_reason": "stop" # Assume stop
                        }],
                        # "usage": {...} # Usage stats are typically not available/meaningful here
                    }
                    return JSONResponse(content=response_payload)

        # Keep outer exception handling for initial connection errors etc.
        except httpx.RequestError as exc:
            print(f"Error requesting Flowith: {exc}")
            raise HTTPException(status_code=503, detail=f"Error connecting to Flowith service: {exc}")
        except HTTPException as http_exc:
            # Re-raise HTTPExceptions (e.g., from status code check or JSON parsing)
            raise http_exc
        except Exception as exc:
            print(f"Unexpected error during Flowith request/processing: {exc}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


# --- Models Endpoint ---
@app.get("/v1/models", response_model=ModelList)
async def list_models(api_key: str = Depends(verify_api_key)): # Protect with existing auth
    """
    Lists the available models based on the models.json mapping.
    Follows the OpenAI API format.
    """
    model_cards = [
        ModelCard(id=model_id) for model_id in model_mappings.keys()
    ]
    return ModelList(data=model_cards)


# --- Optional: Add a root endpoint for health check ---
@app.get("/")
async def root():
    return {"message": "OpenAI to Flowith Proxy is running"}

# --- To run locally (for development) ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
                            fetch_task.cancel()
                        # Send DONE message regardless of success/failure/cancellation
                        yield "data: [DONE]\n\n"

                # The return statement remains the same
                return StreamingResponse(stream_generator(), media_type="text/event-stream")

        except httpx.RequestError as exc:
            print(f"Error requesting Flowith: {exc}")
            raise HTTPException(status_code=503, detail=f"Error connecting to Flowith service: {exc}")
        except HTTPException as http_exc:
            # Re-raise HTTPExceptions (e.g., from status code check or JSON parsing)
            raise http_exc
        except Exception as exc:
            print(f"Unexpected error during Flowith request/processing: {exc}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Internal server error: {exc}")


# --- Models Endpoint ---
@app.get("/v1/models", response_model=ModelList)
async def list_models(api_key: str = Depends(verify_api_key)): # Protect with existing auth
    """
    Lists the available models based on the models.json mapping.
    Follows the OpenAI API format.
    """
    model_cards = [
        ModelCard(id=model_id) for model_id in model_mappings.keys()
    ]
    return ModelList(data=model_cards)


# --- Optional: Add a root endpoint for health check ---
@app.get("/")
async def root():
    return {"message": "OpenAI to Flowith Proxy is running"}

# --- To run locally (for development) ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
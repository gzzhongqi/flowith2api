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
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7,zh-TW;q=0.6,ja;q=0.5',
        'authorization': FLOWITH_AUTH_TOKEN, # Send only the token, no "Bearer " prefix
        'content-type': 'application/json',
        'responsetype': 'stream', # Restore this header
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

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            # Serialize payload manually
            payload_bytes = json.dumps(flowith_payload.dict()).encode('utf-8')

            # Make a non-streaming POST request to Flowith
            response = await client.post(
                FLOWITH_API_URL,
                content=payload_bytes,
                headers=headers,
            )

            # Check status code after receiving the full response
            if response.status_code != 200:
                try:
                    error_detail = response.text # Use .text for non-streaming
                    detail_msg = f"Flowith API Error ({response.status_code}): {error_detail}"
                except Exception:
                    detail_msg = f"Flowith API Error ({response.status_code})"
                raise HTTPException(status_code=response.status_code, detail=detail_msg)

            # Get the plain text response directly
            flowith_text = response.text
            print(f"response preview: {flowith_text[:500]}")
            # 7. Handle response based on *client's* request.stream preference
            if not request.stream:
                # Client wants non-streaming: Construct OpenAI-compatible JSON from plain text
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
                            "content": flowith_text # Use the plain text here
                        },
                        "finish_reason": "stop" # Assume stop
                    }],
                    # "usage": {...} # Usage stats are typically not available/meaningful here
                }
                return JSONResponse(content=response_payload)
            else:
                # Client wants streaming: Implement keep-alive while fetching, then stream response
                async def stream_generator():
                    response_ready_event = asyncio.Event()
                    fetch_task = None
                    flowith_text_local = None # Use a local variable to avoid confusion with outer scope
                    error_occurred = None
                    sse_model_name = request.model # Use the model requested by the client

                    # Coroutine to fetch data and signal completion
                    async def fetch_and_process(event):
                        nonlocal flowith_text_local, error_occurred
                        try:
                            # === Logic to prepare request (model_name, flowith_messages, etc.) ===
                            # These variables are captured from the outer scope:
                            # FLOWITH_API_URL, headers, flowith_payload
                            # The actual request is made here now, not before calling stream_generator
                            async with httpx.AsyncClient(timeout=300.0) as client:
                                payload_bytes = json.dumps(flowith_payload.dict()).encode('utf-8') # Use outer flowith_payload
                                response = await client.post(
                                    FLOWITH_API_URL, headers=headers, content=payload_bytes # Use outer headers
                                )
                            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                            flowith_text_local = response.text # Store in local variable
                            
                        except Exception as e:
                            # print(f"Error fetching from Flowith: {e}") # Optional debug
                            error_occurred = e # Store error to yield later
                        finally:
                            event.set() # Signal completion or failure

                    # Start fetching in the background
                    fetch_task = asyncio.create_task(fetch_and_process(response_ready_event))

                    try:
                        # Send keep-alive chunks while waiting
                        while not response_ready_event.is_set():
                            keep_alive_data = {"id": "chatcmpl-keepalive", "object": "chat.completion.chunk", "created": int(time.time()), "model": sse_model_name, "choices": [{"delta": {"content": ""}, "index": 0, "finish_reason": None}]}
                            yield f"data: {json.dumps(keep_alive_data)}\n\n"
                            # Wait for a short period or until the event is set
                            try:
                                await asyncio.wait_for(response_ready_event.wait(), timeout=3.0)
                            except asyncio.TimeoutError:
                                pass # Timeout means event not set, continue loop

                        # Event is set, check if fetch task had an error
                        if fetch_task.done() and fetch_task.exception():
                            # If fetch_task failed before setting event (shouldn't happen with finally, but check anyway)
                            error_occurred = fetch_task.exception()

                        if error_occurred:
                            # Yield an error chunk (optional, depends on desired behavior)
                            # print(f"Yielding error chunk: {error_occurred}") # Optional debug
                            error_content = f"Error processing request: {type(error_occurred).__name__}"
                            error_chunk = {"id": f"chatcmpl-error-{uuid.uuid4()}", "object": "chat.completion.chunk", "created": int(time.time()), "model": sse_model_name, "choices": [{"delta": {"content": error_content }, "index": 0, "finish_reason": "error"}]} # Use finish_reason 'error' if possible
                            yield f"data: {json.dumps(error_chunk)}\n\n"
                        elif flowith_text_local is not None:
                            # Fetch succeeded, yield content chunks
                            chunk_id = f"chatcmpl-{uuid.uuid4()}"
                            chunk_size = 20
                            full_content = flowith_text_local

                            for i in range(0, len(full_content), chunk_size):
                                content_piece = full_content[i:i + chunk_size]
                                chunk = {
                                    "id": chunk_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": sse_model_name,
                                    "choices": [{"index": 0, "delta": {"content": content_piece}, "finish_reason": None}]
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"

                            # Yield final chunk
                            final_chunk = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": sse_model_name,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                            }
                            yield f"data: {json.dumps(final_chunk)}\n\n"

                    except asyncio.CancelledError:
                         # print("Stream generator cancelled (client disconnected)") # Optional debug
                         raise # Re-raise cancellation
                    finally:
                        # Ensure fetch task is cancelled if generator exits early
                        if fetch_task and not fetch_task.done():
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
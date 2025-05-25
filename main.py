import os
import json
import uuid
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
        role = msg.role
        if is_claude_or_gemini and role == "system":
            # Convert system message to user message for Claude/Gemini via Flowith
            role = "user"
        elif role == "system":
             # If it's a system message but not for Claude/Gemini, Flowith might not support it directly.
             # Option 1: Skip it (might lose context)
             # continue
             # Option 2: Convert to user (might change semantics)
             role = "user"
             # Option 3: Prepend to the next user message (complex)
             # For now, converting to 'user' is a simple approach.
             print(f"Warning: Converting system message to 'user' for model {flowith_model_name}")

        # Ensure only 'user' or 'assistant' roles are sent to Flowith
        if role in ["user", "assistant"]:
             processed_messages.append(FlowithMessage(role=role, content=msg.content))
        # else: # Handle unexpected roles if necessary


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
        'origin': 'https://flowith.net',
        'priority': 'u=1, i',
        'referer': 'https://flowith.net/',
        'responsetype': 'stream', # Added - Was present in -H flags
        'sec-ch-ua': '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'cross-site',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36'
    }

    # 6. Make Asynchronous Request to Flowith
    async with httpx.AsyncClient(timeout=300.0) as client: # Increased timeout for potentially long requests
        try:
            # Always use stream for the request to Flowith
            # Serialize payload manually for content=
            payload_bytes = json.dumps(flowith_payload.dict()).encode('utf-8')
            response_stream = client.stream( # Changed from post to stream
                "POST", # Explicitly set method for stream
                FLOWITH_API_URL,
                content=payload_bytes, # Send serialized bytes
                headers=headers,
            )
            # Need to acquire the stream context
            async with response_stream as response:
                # Check status code *before* attempting to read the stream body
                if response.status_code != 200:
                    # Attempt to read error details from the response body if possible
                    try:
                        error_detail = await response.aread()
                        detail_msg = f"Flowith API Error ({response.status_code}): {error_detail.decode()}"
                    except Exception:
                        detail_msg = f"Flowith API Error ({response.status_code})"
                    # No need to manually close here, 'async with' handles it
                    raise HTTPException(status_code=response.status_code, detail=detail_msg)

                # 7. Handle Flowith Response based on *client's* request.stream preference
                if request.stream:
                    # Client wants streaming: Use StreamingResponse with the helper
                    return StreamingResponse(
                        stream_flowith_response(response), # Pass the response object itself
                        media_type="text/event-stream"
                    )
                else:
                    # Client wants non-streaming: Accumulate the response
                    full_response_bytes = bytearray()
                    try:
                        async for chunk in response.aiter_bytes():
                            full_response_bytes.extend(chunk)
                    except Exception as e:
                         # Handle potential errors during stream reading
                         print(f"Error reading stream from Flowith: {e}")
                         raise HTTPException(status_code=502, detail=f"Error reading stream from Flowith: {e}")
                    finally:
                         # 'async with' ensures the stream is closed
                         pass

                    # Decode the accumulated bytes
                    full_response_text = full_response_bytes.decode('utf-8')

                    # Try to parse as JSON
                    try:
                        response_data = json.loads(full_response_text)
                        from fastapi.responses import JSONResponse # Import locally if not already global
                        return JSONResponse(content=response_data)
                    except json.JSONDecodeError:
                        # If not valid JSON, return as plain text
                        print(f"Warning: Flowith response was not valid JSON. Returning as plain text. Content: {full_response_text[:200]}...") # Log snippet
                        from fastapi.responses import PlainTextResponse # Import locally
                        return PlainTextResponse(content=full_response_text)

        except httpx.RequestError as exc:
            print(f"Error requesting Flowith: {exc}")
            raise HTTPException(status_code=503, detail=f"Error connecting to Flowith service: {exc}")
        except HTTPException as http_exc:
             # Re-raise HTTPExceptions raised during stream processing
             raise http_exc
        except Exception as exc:
             print(f"Unexpected error during Flowith request/processing: {exc}")
             # Log the traceback for debugging
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
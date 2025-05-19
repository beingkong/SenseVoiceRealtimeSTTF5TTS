# LLM (Language Model) Microservice
import os
import logging
import json
import asyncio
import httpx
import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("llm-service")

# Create FastAPI app
app = FastAPI(title="LLM Service", description="Language Model Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:14b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama-service:11434")

# Request models
class Message(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = True

# Initialize clients
openai_client = None
ollama_client = None

@app.on_event("startup")
async def startup_event():
    """Initialize the LLM clients on startup."""
    global openai_client, ollama_client
    
    logger.info(f"Initializing LLM service with provider: {LLM_PROVIDER}")
    
    # Initialize OpenAI client if needed
    if LLM_PROVIDER == "openai" or LLM_PROVIDER == "both":
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            openai.base_url = OPENAI_API_BASE
            openai_client = openai.AsyncClient()
            logger.info("OpenAI client initialized")
        except ImportError:
            logger.warning("OpenAI Python package not installed")
    
    # Initialize Ollama client
    ollama_client = httpx.AsyncClient(base_url=OLLAMA_BASE_URL, timeout=60.0)
    
    # Test Ollama connection
    try:
        response = await ollama_client.get("/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            logger.info(f"Ollama connected successfully. Available models: {[m['name'] for m in models]}")
        else:
            logger.warning(f"Ollama connection test failed: {response.status_code}")
    except Exception as e:
        logger.warning(f"Ollama connection test failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    if ollama_client:
        await ollama_client.aclose()
    if openai_client:
        await openai_client.close()

@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    if LLM_PROVIDER == "openai" and not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI client not initialized")
    if (LLM_PROVIDER == "ollama" or LLM_PROVIDER == "both") and not ollama_client:
        raise HTTPException(status_code=503, detail="Ollama client not initialized")
    return {"status": "ok"}

@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate text using the LLM.
    
    Args:
        request: Generation request parameters
        
    Returns:
        Generated text
    """
    # Determine which provider to use
    provider = LLM_PROVIDER
    model = request.model or LLM_MODEL
    
    # Override provider based on model name if needed
    if model.startswith("gpt-") and provider != "openai":
        provider = "openai"
        logger.info(f"Switching to OpenAI provider for model: {model}")
    elif not model.startswith("gpt-") and provider == "openai":
        provider = "ollama"
        logger.info(f"Switching to Ollama provider for model: {model}")
    
    # Handle streaming
    if request.stream:
        return StreamingResponse(
            generate_stream(provider, model, request),
            media_type="text/event-stream"
        )
    
    # Handle non-streaming
    try:
        if provider == "openai":
            # Use OpenAI
            if not openai_client:
                raise HTTPException(status_code=503, detail="OpenAI client not initialized")
            
            # Convert messages to OpenAI format
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Generate text
            response = await openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False
            )
            
            # Extract text
            text = response.choices[0].message.content
            
            return {"text": text, "model": model}
        else:
            # Use Ollama
            if not ollama_client:
                raise HTTPException(status_code=503, detail="Ollama client not initialized")
            
            # Convert messages to Ollama format
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Prepare request
            ollama_request = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature
                }
            }
            
            if request.max_tokens:
                ollama_request["options"]["num_predict"] = request.max_tokens
            
            # Generate text
            response = await ollama_client.post("/api/chat", json=ollama_request)
            response.raise_for_status()
            
            # Extract text
            result = response.json()
            text = result.get("message", {}).get("content", "")
            
            return {"text": text, "model": model}
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_stream(provider: str, model: str, request: GenerateRequest):
    """
    Stream generated text.
    
    Args:
        provider: LLM provider
        model: Model name
        request: Generation request parameters
        
    Yields:
        Generated text chunks
    """
    try:
        if provider == "openai":
            # Use OpenAI
            if not openai_client:
                yield f"data: {json.dumps({'error': 'OpenAI client not initialized'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Convert messages to OpenAI format
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Generate text
            stream = await openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True
            )
            
            # Stream text
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'text': text})}\n\n"
            
            yield "data: [DONE]\n\n"
        else:
            # Use Ollama
            if not ollama_client:
                yield f"data: {json.dumps({'error': 'Ollama client not initialized'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Convert messages to Ollama format
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Prepare request
            ollama_request = {
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": request.temperature
                }
            }
            
            if request.max_tokens:
                ollama_request["options"]["num_predict"] = request.max_tokens
            
            # Generate text
            async with ollama_client.stream("POST", "/api/chat", json=ollama_request) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            text = data["message"]["content"]
                            yield f"data: {json.dumps({'text': text})}\n\n"
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON: {line}")
            
            yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Error in streaming generation: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

manager = ConnectionManager()

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """
    WebSocket endpoint for text generation.
    
    Args:
        websocket: WebSocket connection
    """
    await manager.connect(websocket)
    
    try:
        # Receive request
        data = await websocket.receive_text()
        request_data = json.loads(data)
        
        # Extract parameters
        messages = request_data.get("messages", [])
        model = request_data.get("model") or LLM_MODEL
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens")
        
        # Determine provider
        provider = LLM_PROVIDER
        
        # Override provider based on model name if needed
        if model.startswith("gpt-") and provider != "openai":
            provider = "openai"
            logger.info(f"Switching to OpenAI provider for model: {model}")
        elif not model.startswith("gpt-") and provider == "openai":
            provider = "ollama"
            logger.info(f"Switching to Ollama provider for model: {model}")
        
        # Generate text
        if provider == "openai":
            # Use OpenAI
            if not openai_client:
                await websocket.send_json({"error": "OpenAI client not initialized"})
                return
            
            # Generate text
            stream = await openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            # Stream text
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    await websocket.send_json({"text": text})
            
            # Send done message
            await websocket.send_json({"done": True})
        else:
            # Use Ollama
            if not ollama_client:
                await websocket.send_json({"error": "Ollama client not initialized"})
                return
            
            # Prepare request
            ollama_request = {
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature
                }
            }
            
            if max_tokens:
                ollama_request["options"]["num_predict"] = max_tokens
            
            # Generate text
            async with ollama_client.stream("POST", "/api/chat", json=ollama_request) as response:
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            text = data["message"]["content"]
                            await websocket.send_json({"text": text})
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON: {line}")
            
            # Send done message
            await websocket.send_json({"done": True})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3004)

#!/usr/bin/env python3
"""
Realtime Voice Chat Server - Part 3: LLM Response Generation and TTS
"""

import io
import base64
from typing import Dict, List, Optional, Any, Union

# System prompt for LLM
SYSTEM_PROMPT = """You are a helpful voice assistant. 
Provide concise, accurate responses to user queries.
Keep your responses conversational but brief, as they will be spoken aloud.
"""

async def generate_response(client_id: str, conversation: List[Dict[str, str]]) -> Optional[str]:
    """
    Generate a response using the LLM service.
    
    Args:
        client_id: Client ID
        conversation: Conversation history
        
    Returns:
        Generated response or None if failed
    """
    try:
        # Prepare conversation with system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(conversation)
        
        # Prepare request data
        request_data = {
            "messages": messages,
            "temperature": 0.7,
            "stream": False
        }
        
        # Send request to LLM service
        async with http_session.post(f"{LLM_SERVICE_URL}/generate", json=request_data) as response:
            if response.status != 200:
                logger.warning(f"LLM service returned status {response.status}")
                return None
            
            # Parse response
            result = await response.json()
            
            # Extract text
            text = result.get("text", "").strip()
            
            # Log response
            if text:
                logger.info(f"LLM response for client {client_id}: {text[:100]}{'...' if len(text) > 100 else ''}")
                
                # Send response to client
                await manager.send_json(client_id, {
                    "type": "response",
                    "text": text
                })
            
            return text
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return None

async def synthesize_speech(client_id: str, text: str) -> None:
    """
    Synthesize speech using the TTS service.
    
    Args:
        client_id: Client ID
        text: Text to synthesize
    """
    try:
        # Prepare request data
        request_data = {
            "text": text,
            "language": "en",
            "return_format": "wav"
        }
        
        # Send request to TTS service
        async with http_session.post(f"{TTS_SERVICE_URL}/synthesize", json=request_data) as response:
            if response.status != 200:
                logger.warning(f"TTS service returned status {response.status}")
                return
            
            # Get audio data
            audio_data = await response.read()
            
            # Get session
            session = session_store.get_session(client_id)
            if not session:
                logger.warning(f"No session found for client {client_id}")
                return
            
            # Add audio to queue
            await session["response_audio_queue"].put(audio_data)
            
            # Log synthesis
            logger.info(f"Synthesized speech for client {client_id}: {len(audio_data)} bytes")
    except Exception as e:
        logger.error(f"Error synthesizing speech: {e}")

# WebSocket endpoint for text chat
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for text chat."""
    # Generate client ID
    client_id = str(uuid.uuid4())
    
    # Accept connection
    await manager.connect(websocket, client_id)
    
    # Create session
    session = session_store.create_session(client_id)
    
    try:
        # Send client ID to client
        await websocket.send_json({"type": "client_id", "client_id": client_id})
        
        # Process incoming messages
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Extract message
            message = data.get("message", "").strip()
            
            # Process message
            if message:
                # Process transcription
                await process_transcription(client_id, message)
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
    finally:
        # Clean up
        manager.disconnect(client_id)
        session_store.delete_session(client_id)

# API endpoint for text chat
class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Chat API endpoint.
    
    Args:
        request: Chat request
        
    Returns:
        Chat response
    """
    # Generate client ID
    client_id = str(uuid.uuid4())
    
    # Create session
    session = session_store.create_session(client_id)
    
    try:
        # Process message
        message = request.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Add user message to conversation
        session["conversation"].append({
            "role": "user",
            "content": message
        })
        
        # Generate response
        response = await generate_response(client_id, session["conversation"])
        
        # If we got a response, add it to conversation
        if response:
            # Add assistant message to conversation
            session["conversation"].append({
                "role": "assistant",
                "content": response
            })
            
            # Return response
            return {"response": response}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate response")
    finally:
        # Clean up
        session_store.delete_session(client_id)

# API endpoint for speech synthesis
class SynthesisRequest(BaseModel):
    text: str
    language: str = "en"
    voice: Optional[str] = None
    return_format: str = "wav"  # "wav" or "base64"

@app.post("/api/synthesize")
async def synthesize(request: SynthesisRequest):
    """
    Speech synthesis API endpoint.
    
    Args:
        request: Synthesis request
        
    Returns:
        Synthesized speech
    """
    try:
        # Prepare request data
        request_data = {
            "text": request.text,
            "language": request.language,
            "voice": request.voice,
            "return_format": request.return_format
        }
        
        # Send request to TTS service
        async with http_session.post(f"{TTS_SERVICE_URL}/synthesize", json=request_data) as response:
            if response.status != 200:
                logger.warning(f"TTS service returned status {response.status}")
                raise HTTPException(status_code=response.status, detail="TTS service error")
            
            # Get content type
            content_type = response.headers.get("Content-Type", "application/octet-stream")
            
            # Get audio data
            audio_data = await response.read()
            
            # Return audio data
            if request.return_format == "base64":
                # Parse JSON response
                result = await response.json()
                return result
            else:
                # Return WAV data
                return Response(content=audio_data, media_type=content_type)
    except Exception as e:
        logger.error(f"Error synthesizing speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Periodic tasks
async def periodic_tasks():
    """Run periodic tasks."""
    while True:
        try:
            # Clean up old sessions
            session_store.cleanup_old_sessions()
            
            # Wait for next run
            await asyncio.sleep(3600)  # Run every hour
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic tasks: {e}")
            await asyncio.sleep(60)  # Wait a bit before retrying

# Start periodic tasks
@app.on_event("startup")
async def start_periodic_tasks():
    """Start periodic tasks."""
    asyncio.create_task(periodic_tasks())

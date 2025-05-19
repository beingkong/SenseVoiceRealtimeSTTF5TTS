# STT (Speech-to-Text) Microservice using RealtimeSTT
import os
import logging
import numpy as np
import torch
import asyncio
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, Query, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("stt-service")

# Create FastAPI app
app = FastAPI(title="STT Service", description="Speech-to-Text Service using RealtimeSTT")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANGUAGE = os.getenv("LANGUAGE", "en")
RETURN_TIMESTAMPS = os.getenv("RETURN_TIMESTAMPS", "false").lower() == "true"

# Initialize STT model
stt_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the STT model on startup."""
    global stt_model
    
    logger.info(f"Initializing RealtimeSTT model on {DEVICE}")
    
    try:
        # Try to import RealtimeSTT (this will fail if the package is not installed)
        try:
            from realtimestt import RealtimeSTT
            
            # Initialize RealtimeSTT
            stt_model = RealtimeSTT(
                device=DEVICE,
                language=LANGUAGE,
                return_timestamps=RETURN_TIMESTAMPS
            )
            
            logger.info(f"RealtimeSTT model loaded successfully on {DEVICE}")
        except ImportError:
            # RealtimeSTT package not found, use our own implementation
            logger.info("RealtimeSTT package not found, using Hugging Face Transformers")
            raise ImportError("RealtimeSTT package not found")
            
    except Exception as e:
        logger.error(f"Error loading RealtimeSTT model: {e}")
        logger.info("Falling back to Hugging Face Transformers")
        
        # Fallback to Hugging Face Transformers
        try:
            from transformers import pipeline
            
            logger.info("Falling back to Hugging Face Transformers pipeline")
            
            # Initialize pipeline
            stt_model = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                device=DEVICE
            )
            
            # Wrap pipeline in a class with the same interface
            class WhisperWrapper:
                def __init__(self, pipeline):
                    self.pipeline = pipeline
                
                def transcribe(self, audio, language="en"):
                    """
                    Transcribe audio to text.
                    
                    Args:
                        audio: Audio data as numpy array
                        language: Language code
                        
                    Returns:
                        Dictionary with transcription results
                    """
                    result = self.pipeline(
                        audio,
                        language=language
                    )
                    
                    return {
                        "text": result["text"],
                        "language": language
                    }
            
            stt_model = WhisperWrapper(stt_model)
            logger.info("Whisper fallback initialized")
        except Exception as e:
            logger.error(f"Error initializing fallback STT model: {e}")
            raise

@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    if stt_model is None:
        raise HTTPException(status_code=503, detail="STT model not initialized")
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Query("en", description="Language code")
):
    """
    Transcribe audio to text.
    
    Args:
        file: Audio file (raw PCM)
        language: Language code
        
    Returns:
        Dictionary with transcription results
    """
    try:
        # Read audio data
        audio_bytes = await file.read()
        
        # Convert to numpy array (assuming 16-bit PCM)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Convert to float32 and normalize
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Transcribe audio
        result = stt_model.transcribe(audio_data, language=language)
        
        # Format result
        transcription = {
            "text": result.get("text", ""),
            "language": language,
            "is_final": True
        }
        
        # Add timestamps if available
        if "timestamps" in result:
            transcription["timestamps"] = result["timestamps"]
            
        return transcription
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

# Buffer class for streaming audio
class AudioBuffer:
    def __init__(self, sample_rate=16000, max_buffer_size=16000*30):
        self.buffer = np.array([], dtype=np.float32)
        self.sample_rate = sample_rate
        self.max_buffer_size = max_buffer_size
        
    def add_chunk(self, chunk: np.ndarray):
        """Add a chunk to the buffer."""
        self.buffer = np.append(self.buffer, chunk)
        
        # Trim buffer if it gets too large
        if len(self.buffer) > self.max_buffer_size:
            self.buffer = self.buffer[-self.max_buffer_size:]
            
    def get_buffer(self) -> np.ndarray:
        """Get the current buffer."""
        return self.buffer.copy()
        
    def clear(self):
        """Clear the buffer."""
        self.buffer = np.array([], dtype=np.float32)

@app.websocket("/ws/transcribe")
async def websocket_transcribe(
    websocket: WebSocket,
    language: str = Query("en", description="Language code")
):
    """
    WebSocket endpoint for streaming transcription.
    
    Args:
        websocket: WebSocket connection
        language: Language code
    """
    await manager.connect(websocket)
    
    # Create audio buffer
    audio_buffer = AudioBuffer(sample_rate=SAMPLE_RATE)
    
    # Create task for periodic transcription
    transcription_task = None
    
    async def transcribe_periodically():
        """Periodically transcribe the audio buffer."""
        try:
            while True:
                # Wait a bit to accumulate audio
                await asyncio.sleep(0.5)
                
                # Get current buffer
                buffer = audio_buffer.get_buffer()
                
                # Skip if buffer is empty
                if len(buffer) < SAMPLE_RATE:
                    continue
                
                # Transcribe audio
                result = stt_model.transcribe(buffer, language=language)
                
                # Format result
                transcription = {
                    "text": result.get("text", ""),
                    "language": language,
                    "is_final": False
                }
                
                # Send result to client
                await websocket.send_text(json.dumps(transcription))
        except asyncio.CancelledError:
            logger.info("Transcription task cancelled")
        except Exception as e:
            logger.error(f"Error in transcription task: {e}")
    
    try:
        # Start transcription task
        transcription_task = asyncio.create_task(transcribe_periodically())
        
        # Process incoming audio chunks
        while True:
            # Receive audio chunk
            chunk = await websocket.receive_bytes()
            
            # Check if this is a signal to end the stream
            if not chunk:
                # Send final transcription
                buffer = audio_buffer.get_buffer()
                if len(buffer) > 0:
                    result = stt_model.transcribe(buffer, language=language)
                    
                    # Format result
                    transcription = {
                        "text": result.get("text", ""),
                        "language": language,
                        "is_final": True
                    }
                    
                    # Send result to client
                    await websocket.send_text(json.dumps(transcription))
                break
            
            # Convert to numpy array (assuming 16-bit PCM)
            audio_data = np.frombuffer(chunk, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Add to buffer
            audio_buffer.add_chunk(audio_data)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
    finally:
        # Clean up
        manager.disconnect(websocket)
        if transcription_task:
            transcription_task.cancel()
            try:
                await transcription_task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3002)

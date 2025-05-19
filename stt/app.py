# STT (Speech-to-Text) Microservice
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
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("stt-service")

# Create FastAPI app
app = FastAPI(title="STT Service", description="Speech-to-Text Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
MODEL_NAME = os.getenv("STT_MODEL", "openai/whisper-base")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_LENGTH_S = float(os.getenv("CHUNK_LENGTH_S", "30.0"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
RETURN_TIMESTAMPS = os.getenv("RETURN_TIMESTAMPS", "false").lower() == "true"

# Initialize STT model
stt_pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize the STT model on startup."""
    global stt_pipeline
    
    logger.info(f"Initializing STT model: {MODEL_NAME} on {DEVICE}")
    
    try:
        # Load model and processor
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(DEVICE)
        
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        
        # Create pipeline
        stt_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=CHUNK_LENGTH_S,
            batch_size=BATCH_SIZE,
            return_timestamps=RETURN_TIMESTAMPS,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device=DEVICE,
        )
        
        logger.info(f"STT model loaded successfully on {DEVICE}")
    except Exception as e:
        logger.error(f"Error loading STT model: {e}")
        raise

@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    if stt_pipeline is None:
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
        result = stt_pipeline(
            audio_data,
            sampling_rate=SAMPLE_RATE,
            generate_kwargs={"language": language} if "whisper" in MODEL_NAME.lower() else None,
        )
        
        # Format result
        transcription = {
            "text": result["text"],
            "language": language,
            "is_final": True
        }
        
        # Add timestamps if available
        if "chunks" in result:
            transcription["chunks"] = result["chunks"]
            
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
                result = stt_pipeline(
                    buffer,
                    sampling_rate=SAMPLE_RATE,
                    generate_kwargs={"language": language} if "whisper" in MODEL_NAME.lower() else None,
                )
                
                # Format result
                transcription = {
                    "text": result["text"],
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
                    result = stt_pipeline(
                        buffer,
                        sampling_rate=SAMPLE_RATE,
                        generate_kwargs={"language": language} if "whisper" in MODEL_NAME.lower() else None,
                    )
                    
                    # Format result
                    transcription = {
                        "text": result["text"],
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

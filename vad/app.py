# VAD (Voice Activity Detection) Microservice
import os
import logging
import numpy as np
import torch
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vad-service")

# Create FastAPI app
app = FastAPI(title="VAD Service", description="Voice Activity Detection Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
VAD_MODEL = os.getenv("VAD_MODEL", "silero")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize VAD model
vad_model = None
get_speech_timestamps = None

# Request model
class VADRequest(BaseModel):
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = 30.0
    min_silence_duration_ms: int = 100
    window_size_samples: int = 512
    speech_pad_ms: int = 30

@app.on_event("startup")
async def startup_event():
    """Initialize the VAD model on startup."""
    global vad_model, get_speech_timestamps
    
    logger.info(f"Initializing VAD model: {VAD_MODEL} on {DEVICE}")
    
    try:
        if VAD_MODEL == "silero":
            # Load Silero VAD
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False
            )
            
            # Get utils
            (get_speech_timestamps, _, _, _, _) = utils
            
            # Move model to device
            vad_model = model.to(DEVICE)
            
            logger.info(f"Silero VAD model loaded successfully on {DEVICE}")
        else:
            # Fallback to a simple energy-based VAD
            class EnergyVAD:
                def __init__(self):
                    pass
                
                def __call__(self, audio, threshold=0.5):
                    """
                    Simple energy-based VAD.
                    
                    Args:
                        audio: Audio data as tensor
                        threshold: Energy threshold
                        
                    Returns:
                        Speech probability
                    """
                    # Convert to numpy if tensor
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().numpy()
                    
                    # Calculate energy
                    energy = np.abs(audio)
                    
                    # Calculate mean energy
                    mean_energy = np.mean(energy)
                    
                    # Calculate speech probability
                    speech_prob = min(1.0, mean_energy / threshold)
                    
                    return speech_prob
            
            vad_model = EnergyVAD()
            
            # Define get_speech_timestamps function
            def energy_speech_timestamps(audio, threshold=0.5, sampling_rate=16000, 
                                        min_speech_duration_ms=250, max_speech_duration_s=float('inf'),
                                        min_silence_duration_ms=100, window_size_samples=512,
                                        speech_pad_ms=30):
                """
                Get speech timestamps using energy-based VAD.
                
                Args:
                    audio: Audio data as tensor or numpy array
                    threshold: Energy threshold
                    sampling_rate: Audio sampling rate
                    min_speech_duration_ms: Minimum speech duration in ms
                    max_speech_duration_s: Maximum speech duration in seconds
                    min_silence_duration_ms: Minimum silence duration in ms
                    window_size_samples: Window size in samples
                    speech_pad_ms: Speech padding in ms
                    
                Returns:
                    List of speech timestamps
                """
                # Convert to numpy if tensor
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                
                # Calculate energy
                energy = np.abs(audio)
                
                # Apply threshold
                speech = energy > threshold
                
                # Convert to samples
                min_speech_samples = int(min_speech_duration_ms * sampling_rate / 1000)
                max_speech_samples = int(max_speech_duration_s * sampling_rate)
                min_silence_samples = int(min_silence_duration_ms * sampling_rate / 1000)
                speech_pad_samples = int(speech_pad_ms * sampling_rate / 1000)
                
                # Find speech segments
                speech_segments = []
                in_speech = False
                speech_start = 0
                
                for i in range(0, len(speech), window_size_samples):
                    window = speech[i:i+window_size_samples]
                    if np.mean(window) > 0.5 and not in_speech:
                        # Start of speech
                        in_speech = True
                        speech_start = max(0, i - speech_pad_samples)
                    elif np.mean(window) <= 0.5 and in_speech:
                        # End of speech
                        speech_end = min(len(speech), i + speech_pad_samples)
                        
                        # Check if speech segment is long enough
                        if speech_end - speech_start >= min_speech_samples:
                            # Check if speech segment is not too long
                            if speech_end - speech_start <= max_speech_samples:
                                speech_segments.append({
                                    "start": speech_start,
                                    "end": speech_end
                                })
                            else:
                                # Split long speech segment
                                for j in range(speech_start, speech_end, max_speech_samples):
                                    segment_end = min(j + max_speech_samples, speech_end)
                                    speech_segments.append({
                                        "start": j,
                                        "end": segment_end
                                    })
                        
                        in_speech = False
                
                # Handle case where audio ends during speech
                if in_speech:
                    speech_end = len(speech)
                    
                    # Check if speech segment is long enough
                    if speech_end - speech_start >= min_speech_samples:
                        # Check if speech segment is not too long
                        if speech_end - speech_start <= max_speech_samples:
                            speech_segments.append({
                                "start": speech_start,
                                "end": speech_end
                            })
                        else:
                            # Split long speech segment
                            for j in range(speech_start, speech_end, max_speech_samples):
                                segment_end = min(j + max_speech_samples, speech_end)
                                speech_segments.append({
                                    "start": j,
                                    "end": segment_end
                                })
                
                return speech_segments
            
            get_speech_timestamps = energy_speech_timestamps
            
            logger.info("Energy-based VAD initialized")
    except Exception as e:
        logger.error(f"Error loading VAD model: {e}")
        raise

@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    if vad_model is None:
        raise HTTPException(status_code=503, detail="VAD model not initialized")
    return {"status": "ok"}

@app.post("/detect")
async def detect_voice_activity(
    file: UploadFile = File(...),
    params: Optional[VADRequest] = None
):
    """
    Detect voice activity in audio.
    
    Args:
        file: Audio file (raw PCM)
        params: VAD parameters
        
    Returns:
        Dictionary with speech timestamps
    """
    try:
        # Use default parameters if not provided
        if params is None:
            params = VADRequest()
        
        # Read audio data
        audio_bytes = await file.read()
        
        # Convert to numpy array (assuming 16-bit PCM)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Convert to float32 and normalize
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio_data)
        
        if VAD_MODEL == "silero":
            # Get speech timestamps
            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                vad_model,
                threshold=params.threshold,
                sampling_rate=SAMPLE_RATE,
                min_speech_duration_ms=params.min_speech_duration_ms,
                max_speech_duration_s=params.max_speech_duration_s,
                min_silence_duration_ms=params.min_silence_duration_ms,
                window_size_samples=params.window_size_samples,
                speech_pad_ms=params.speech_pad_ms
            )
        else:
            # Use energy-based VAD
            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                threshold=params.threshold,
                sampling_rate=SAMPLE_RATE,
                min_speech_duration_ms=params.min_speech_duration_ms,
                max_speech_duration_s=params.max_speech_duration_s,
                min_silence_duration_ms=params.min_silence_duration_ms,
                window_size_samples=params.window_size_samples,
                speech_pad_ms=params.speech_pad_ms
            )
        
        # Convert timestamps to seconds
        speech_segments = []
        for ts in speech_timestamps:
            speech_segments.append({
                "start": ts["start"] / SAMPLE_RATE,
                "end": ts["end"] / SAMPLE_RATE,
                "duration": (ts["end"] - ts["start"]) / SAMPLE_RATE
            })
        
        return {
            "speech_segments": speech_segments,
            "sample_rate": SAMPLE_RATE,
            "audio_duration": len(audio_data) / SAMPLE_RATE
        }
    except Exception as e:
        logger.error(f"Error detecting voice activity: {e}")
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
    def __init__(self, sample_rate=16000, max_buffer_size=16000*5):
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

@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket endpoint for streaming voice activity detection.
    
    Args:
        websocket: WebSocket connection
    """
    await manager.connect(websocket)
    
    # Create audio buffer
    audio_buffer = AudioBuffer(sample_rate=SAMPLE_RATE)
    
    # Create task for periodic VAD
    vad_task = None
    
    # Get VAD parameters
    try:
        params_json = await websocket.receive_text()
        params = VADRequest(**json.loads(params_json))
    except Exception as e:
        logger.error(f"Error parsing VAD parameters: {e}")
        params = VADRequest()
    
    async def detect_periodically():
        """Periodically detect voice activity in the buffer."""
        try:
            while True:
                # Wait a bit to accumulate audio
                await asyncio.sleep(0.1)
                
                # Get current buffer
                buffer = audio_buffer.get_buffer()
                
                # Skip if buffer is empty
                if len(buffer) < params.window_size_samples:
                    continue
                
                # Convert to tensor
                audio_tensor = torch.tensor(buffer)
                
                if VAD_MODEL == "silero":
                    # Get speech timestamps
                    speech_timestamps = get_speech_timestamps(
                        audio_tensor,
                        vad_model,
                        threshold=params.threshold,
                        sampling_rate=SAMPLE_RATE,
                        min_speech_duration_ms=params.min_speech_duration_ms,
                        max_speech_duration_s=params.max_speech_duration_s,
                        min_silence_duration_ms=params.min_silence_duration_ms,
                        window_size_samples=params.window_size_samples,
                        speech_pad_ms=params.speech_pad_ms
                    )
                else:
                    # Use energy-based VAD
                    speech_timestamps = get_speech_timestamps(
                        audio_tensor,
                        threshold=params.threshold,
                        sampling_rate=SAMPLE_RATE,
                        min_speech_duration_ms=params.min_speech_duration_ms,
                        max_speech_duration_s=params.max_speech_duration_s,
                        min_silence_duration_ms=params.min_silence_duration_ms,
                        window_size_samples=params.window_size_samples,
                        speech_pad_ms=params.speech_pad_ms
                    )
                
                # Convert timestamps to seconds
                speech_segments = []
                for ts in speech_timestamps:
                    speech_segments.append({
                        "start": ts["start"] / SAMPLE_RATE,
                        "end": ts["end"] / SAMPLE_RATE,
                        "duration": (ts["end"] - ts["start"]) / SAMPLE_RATE
                    })
                
                # Send result to client
                await websocket.send_text(json.dumps({
                    "speech_segments": speech_segments,
                    "buffer_duration": len(buffer) / SAMPLE_RATE
                }))
        except asyncio.CancelledError:
            logger.info("VAD task cancelled")
        except Exception as e:
            logger.error(f"Error in VAD task: {e}")
    
    try:
        # Start VAD task
        vad_task = asyncio.create_task(detect_periodically())
        
        # Process incoming audio chunks
        while True:
            # Receive audio chunk
            chunk = await websocket.receive_bytes()
            
            # Check if this is a signal to end the stream
            if not chunk:
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
        if vad_task:
            vad_task.cancel()
            try:
                await vad_task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)

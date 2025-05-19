# VAD (Voice Activity Detection) Microservice using SenseVoice
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
import soundfile as sf
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vad-service")

# Create FastAPI app
app = FastAPI(title="VAD Service", description="Voice Activity Detection Service using SenseVoice")

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

# Initialize VAD model
vad_model = None

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
    """Initialize the SenseVoice VAD model on startup."""
    global vad_model
    
    logger.info(f"Initializing SenseVoice VAD model on {DEVICE}")
    
    try:
        # Try to import SenseVoice (this will fail if the package is not installed)
        try:
            # Try to import the sensevoice-onnx package
            try:
                # Print the Python path to debug
                import sys
                logger.info(f"Python path: {sys.path}")
    
                
                # Initialize SenseVoice VAD with ONNX backend
                device_id = -1 if DEVICE == "cpu" else 0  # Use -1 for CPU, GPU index otherwise
                
                # Import the SenseVoice class directly from the module
                try:
                    from sensevoice.sense_voice import SenseVoiceInferenceSession
                    import os
                    
                    # Use the exact file paths from the logs
                    embedding_model_file = "/usr/local/lib/python3.10/dist-packages/sensevoice/resource/embedding.npy"
                    encoder_model_file = "/usr/local/lib/python3.10/dist-packages/sensevoice/resource/sense-voice-encoder.onnx"
                    bpe_model_file = "/usr/local/lib/python3.10/dist-packages/sensevoice/resource/chn_jpn_yue_eng_ko_spectok.bpe.model"
                    
                    logger.info(f"SenseVoice model files:")
                    logger.info(f"  Embedding: {embedding_model_file}")
                    logger.info(f"  Encoder: {encoder_model_file}")
                    logger.info(f"  BPE: {bpe_model_file}")
                    
                    # Check if files exist
                    if not os.path.exists(embedding_model_file):
                        logger.warning(f"Embedding model file not found: {embedding_model_file}")
                    else:
                        logger.info(f"Embedding model file exists: {embedding_model_file}")
                    
                    if not os.path.exists(encoder_model_file):
                        logger.warning(f"Encoder model file not found: {encoder_model_file}")
                    else:
                        logger.info(f"Encoder model file exists: {encoder_model_file}")
                    
                    if not os.path.exists(bpe_model_file):
                        logger.warning(f"BPE model file not found: {bpe_model_file}")
                        # Try to find it in the resource directory
                        import sensevoice
                        sensevoice_dir = os.path.dirname(sensevoice.__file__)
                        resource_dir = os.path.join(sensevoice_dir, "resource")
                        if os.path.exists(resource_dir):
                            for file in os.listdir(resource_dir):
                                if file.endswith(".model"):
                                    bpe_model_file = os.path.join(resource_dir, file)
                                    logger.info(f"Found potential BPE model file: {bpe_model_file}")
                                    break
                    else:
                        logger.info(f"BPE model file exists: {bpe_model_file}")
                    
                    # Initialize with the required parameters
                    vad_model = SenseVoiceInferenceSession(
                        embedding_model_file=embedding_model_file,
                        encoder_model_file=encoder_model_file,
                        bpe_model_file=bpe_model_file
                    )
                    logger.info(f"SenseVoice-ONNX VAD model loaded successfully")
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not import SenseVoice class: {e}")
                    logger.info("Using sensevoice CLI as a wrapper")
                    
                    # Create a wrapper class that uses the CLI
                    class SenseVoiceCLIWrapper:
                        def __init__(self, device=-1):
                            self.device = device
                            logger.info(f"Initialized SenseVoice CLI wrapper with device {device}")
                        
                        def detect_speech(self, audio, **kwargs):
                            """Use the sensevoice CLI to detect speech."""
                            import tempfile
                            import subprocess
                            import json
                            import numpy as np
                            
                            # Save audio to a temporary file
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                                temp_filename = f.name
                                import soundfile as sf
                                sf.write(temp_filename, audio, 16000)
                            
                            # Run the sensevoice CLI
                            try:
                                cmd = ["sensevoice", "--audio", temp_filename]
                                if self.device >= 0:
                                    cmd.extend(["-d", str(self.device)])
                                
                                logger.info(f"Running command: {' '.join(cmd)}")
                                result = subprocess.run(cmd, capture_output=True, text=True)
                                logger.info(f"SenseVoice CLI output: {result.stdout}")
                                
                                # Parse the output to find speech segments
                                # Example output: [0.61s - 5.53s] <|zh|><|NEUTRAL|><|Speech|><|woitn|>text
                                import re
                                segments = []
                                
                                # Look for time ranges like [0.61s - 5.53s]
                                matches = re.findall(r'\[(\d+\.\d+)s - (\d+\.\d+)s\]', result.stdout)
                                for start_str, end_str in matches:
                                    start = float(start_str)
                                    end = float(end_str)
                                    
                                    # Convert to samples
                                    start_sample = int(start * 16000)
                                    end_sample = int(end * 16000)
                                    
                                    segments.append({
                                        "start": start_sample,
                                        "end": end_sample
                                    })
                                
                                # Clean up
                                import os
                                os.unlink(temp_filename)
                                
                                return segments
                            except Exception as e:
                                logger.error(f"Error running SenseVoice CLI: {e}")
                                # Clean up
                                import os
                                os.unlink(temp_filename)
                                # Return empty list
                                return []
                    
                    vad_model = SenseVoiceCLIWrapper(device=device_id)
                    logger.info("SenseVoice CLI wrapper initialized")
            except ImportError as e:
                # Try the original SenseVoice package as a last resort
                logger.info(f"SenseVoice-ONNX package not found: {e}")
                logger.info("Trying original SenseVoice")
                try:
                    from sensevoice import SenseVoice
                    
                    # Initialize SenseVoice VAD
                    vad_model = SenseVoice(device=DEVICE)
                    
                    logger.info(f"SenseVoice VAD model loaded successfully on {DEVICE}")
                except ImportError as e:
                    logger.info(f"Original SenseVoice package not found: {e}")
                    raise ImportError("SenseVoice packages not found")
        except ImportError:
            # SenseVoice package not found, use our own implementation
            logger.info("SenseVoice packages not found, using built-in implementation")
            raise ImportError("SenseVoice packages not found")
            
    except Exception as e:
        logger.error(f"Error loading SenseVoice VAD model: {e}")
        logger.info("Using energy-based VAD implementation")
        
        # Fallback to a simple energy-based VAD
        class EnergyVAD:
            def __init__(self):
                pass
            
            def detect_speech(self, audio, threshold=0.5, **kwargs):
                """
                Simple energy-based VAD.
                
                Args:
                    audio: Audio data as numpy array
                    threshold: Energy threshold
                    
                Returns:
                    List of speech segments
                """
                # Calculate energy
                energy = np.abs(audio)
                
                # Apply threshold
                speech = energy > threshold
                
                # Find speech segments
                speech_segments = []
                in_speech = False
                speech_start = 0
                
                window_size = kwargs.get('window_size_samples', 512)
                min_speech_samples = int(kwargs.get('min_speech_duration_ms', 250) * SAMPLE_RATE / 1000)
                max_speech_samples = int(kwargs.get('max_speech_duration_s', 30.0) * SAMPLE_RATE)
                min_silence_samples = int(kwargs.get('min_silence_duration_ms', 100) * SAMPLE_RATE / 1000)
                speech_pad_samples = int(kwargs.get('speech_pad_ms', 30) * SAMPLE_RATE / 1000)
                
                for i in range(0, len(speech), window_size):
                    window = speech[i:i+window_size]
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
        
        vad_model = EnergyVAD()
        logger.info("Energy-based VAD initialized as fallback")

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
        
        # Detect speech using SenseVoice
        if hasattr(vad_model, 'detect_speech'):
            # SenseVoice API
            speech_timestamps = vad_model.detect_speech(
                audio_data,
                threshold=params.threshold,
                min_speech_duration_ms=params.min_speech_duration_ms,
                max_speech_duration_s=params.max_speech_duration_s,
                min_silence_duration_ms=params.min_silence_duration_ms,
                window_size_samples=params.window_size_samples,
                speech_pad_ms=params.speech_pad_ms
            )
        else:
            # Fallback to energy-based VAD
            speech_timestamps = vad_model.detect_speech(
                audio_data,
                threshold=params.threshold,
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
                
                # Detect speech using SenseVoice
                if hasattr(vad_model, 'detect_speech'):
                    # SenseVoice API
                    speech_timestamps = vad_model.detect_speech(
                        buffer,
                        threshold=params.threshold,
                        min_speech_duration_ms=params.min_speech_duration_ms,
                        max_speech_duration_s=params.max_speech_duration_s,
                        min_silence_duration_ms=params.min_silence_duration_ms,
                        window_size_samples=params.window_size_samples,
                        speech_pad_ms=params.speech_pad_ms
                    )
                else:
                    # Fallback to energy-based VAD
                    speech_timestamps = vad_model.detect_speech(
                        buffer,
                        threshold=params.threshold,
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

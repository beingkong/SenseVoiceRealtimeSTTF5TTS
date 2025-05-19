# TTS (Text-to-Speech) Microservice using F5 TTS
import os
import io
import time
import logging
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("tts-service")

# Create FastAPI app
app = FastAPI(title="TTS Service", description="Text-to-Speech Service using F5 TTS")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
TTS_MODEL = os.getenv("TTS_MODEL", "f5/default")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "24000"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize TTS model
tts_model = None

# Request model
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    language: str = "en"
    speed: float = 1.0
    return_format: str = "wav"  # "wav" or "base64"

@app.on_event("startup")
async def startup_event():
    """Initialize the TTS model on startup."""
    global tts_model
    
    logger.info(f"Initializing F5 TTS model: {TTS_MODEL} on {DEVICE}")
    
    try:
        # Try to import F5 TTS (this will fail if the package is not installed)
        try:
            from f5tts import F5TTS
            
            # Initialize F5 TTS
            tts_model = F5TTS(
                model=TTS_MODEL,
                device=DEVICE
            )
            
            logger.info(f"F5 TTS model loaded successfully on {DEVICE}")
        except ImportError:
            # F5 TTS package not found, use our own implementation
            logger.info("F5 TTS package not found, using gTTS")
            raise ImportError("F5 TTS package not found")
            
    except Exception as e:
        logger.error(f"Error loading F5 TTS model: {e}")
        logger.info("Falling back to gTTS")
        
        # Fallback to a simple TTS engine
        from gtts import gTTS
        
        # Create a wrapper for gTTS
        class GTTS:
            def __init__(self):
                pass
            
            def synthesize(self, text, voice=None, language="en", speed=1.0):
                """
                Convert text to speech.
                
                Args:
                    text: Text to convert
                    voice: Speaker voice (ignored)
                    language: Language code
                    speed: Speed factor (ignored)
                    
                Returns:
                    Audio data as numpy array and sample rate
                """
                # Create a BytesIO object
                mp3_fp = io.BytesIO()
                
                # Create gTTS object
                tts = gTTS(text=text, lang=language, slow=False)
                
                # Save to BytesIO
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                
                # Convert to WAV using pydub
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_mp3(mp3_fp)
                    wav_fp = io.BytesIO()
                    audio.export(wav_fp, format="wav")
                    wav_fp.seek(0)
                    
                    # Convert to numpy array
                    import wave
                    with wave.open(wav_fp, "rb") as wav_file:
                        # Get audio parameters
                        n_channels = wav_file.getnchannels()
                        sample_width = wav_file.getsampwidth()
                        framerate = wav_file.getframerate()
                        n_frames = wav_file.getnframes()
                        
                        # Read audio data
                        audio_data = wav_file.readframes(n_frames)
                        
                        # Convert to numpy array
                        if sample_width == 2:
                            dtype = np.int16
                        elif sample_width == 4:
                            dtype = np.int32
                        else:
                            dtype = np.uint8
                            
                        audio_array = np.frombuffer(audio_data, dtype=dtype)
                        
                        # Reshape for multiple channels
                        if n_channels > 1:
                            audio_array = audio_array.reshape(-1, n_channels)
                        
                        return audio_array, framerate
                except ImportError:
                    logger.warning("pydub not installed, returning raw MP3 data")
                    return mp3_fp.read(), 0
        
        tts_model = GTTS()
        logger.info("gTTS model initialized as fallback")

@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not initialized")
    return {"status": "ok"}

@app.post("/synthesize")
async def synthesize(request: Request, tts_request: TTSRequest):
    """
    Synthesize text to speech.
    
    Args:
        tts_request: TTS request parameters
        
    Returns:
        Audio data as WAV or base64-encoded string
    """
    try:
        start_time = time.time()
        
        # Extract parameters
        text = tts_request.text
        voice = tts_request.voice
        language = tts_request.language
        speed = tts_request.speed
        return_format = tts_request.return_format
        
        logger.info(f"Synthesizing text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Synthesize speech
        if hasattr(tts_model, 'synthesize'):
            # Use F5 TTS
            audio_data, sample_rate = tts_model.synthesize(
                text=text,
                voice=voice,
                language=language,
                speed=speed
            )
            
            # Convert to bytes
            wav_bytes = io.BytesIO()
            import scipy.io.wavfile as wavfile
            wavfile.write(wav_bytes, sample_rate, audio_data)
            wav_bytes.seek(0)
            audio_data = wav_bytes.read()
        else:
            # Use fallback TTS
            audio_data, sample_rate = tts_model.synthesize(
                text=text,
                voice=voice,
                language=language,
                speed=speed
            )
            
            # Convert to bytes if needed
            if isinstance(audio_data, np.ndarray):
                wav_bytes = io.BytesIO()
                import scipy.io.wavfile as wavfile
                wavfile.write(wav_bytes, sample_rate or SAMPLE_RATE, audio_data)
                wav_bytes.seek(0)
                audio_data = wav_bytes.read()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Estimate duration (assuming 16-bit PCM mono)
        if isinstance(audio_data, bytes) and return_format == "wav":
            # Rough estimate: bytes / (sample_rate * bytes_per_sample)
            duration = len(audio_data) / ((sample_rate or SAMPLE_RATE) * 2)
        else:
            duration = 0
        
        # Return as requested format
        if return_format == "base64":
            # Return as base64-encoded string
            base64_data = base64.b64encode(audio_data).decode("utf-8")
            return {
                "audio": base64_data,
                "duration": duration,
                "processing_time": processing_time
            }
        else:
            # Return as WAV
            response = Response(content=audio_data, media_type="audio/wav")
            response.headers["X-Duration"] = str(duration)
            response.headers["X-Processing-Time"] = str(processing_time)
            return response
    except Exception as e:
        logger.error(f"Error synthesizing speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize/stream")
async def synthesize_stream(tts_request: TTSRequest):
    """
    Stream synthesized speech.
    
    Args:
        tts_request: TTS request parameters
        
    Returns:
        Streaming response with audio data
    """
    try:
        # Extract parameters
        text = tts_request.text
        voice = tts_request.voice
        language = tts_request.language
        speed = tts_request.speed
        
        logger.info(f"Streaming synthesis for text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Define generator function for streaming
        async def generate():
            try:
                # Synthesize speech
                if hasattr(tts_model, 'synthesize'):
                    # Use F5 TTS
                    audio_data, sample_rate = tts_model.synthesize(
                        text=text,
                        voice=voice,
                        language=language,
                        speed=speed
                    )
                    
                    # Convert to bytes
                    wav_bytes = io.BytesIO()
                    import scipy.io.wavfile as wavfile
                    wavfile.write(wav_bytes, sample_rate, audio_data)
                    wav_bytes.seek(0)
                    
                    # Yield audio data in chunks
                    chunk_size = 4096  # 4KB chunks
                    while True:
                        chunk = wav_bytes.read(chunk_size)
                        if not chunk:
                            break
                        yield chunk
                else:
                    # Use fallback TTS
                    audio_data, sample_rate = tts_model.synthesize(
                        text=text,
                        voice=voice,
                        language=language,
                        speed=speed
                    )
                    
                    # Convert to bytes if needed
                    if isinstance(audio_data, np.ndarray):
                        wav_bytes = io.BytesIO()
                        import scipy.io.wavfile as wavfile
                        wavfile.write(wav_bytes, sample_rate or SAMPLE_RATE, audio_data)
                        wav_bytes.seek(0)
                        
                        # Yield audio data in chunks
                        chunk_size = 4096  # 4KB chunks
                        while True:
                            chunk = wav_bytes.read(chunk_size)
                            if not chunk:
                                break
                            yield chunk
                    else:
                        # Audio data is already bytes
                        yield audio_data
            except Exception as e:
                logger.error(f"Error in streaming synthesis: {e}")
                raise
        
        # Return streaming response
        return StreamingResponse(generate(), media_type="audio/wav")
    except Exception as e:
        logger.error(f"Error setting up streaming synthesis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3003)

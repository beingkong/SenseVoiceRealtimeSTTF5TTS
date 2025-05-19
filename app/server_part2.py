#!/usr/bin/env python3
"""
Realtime Voice Chat Server - Part 2: Audio Processing and Speech Recognition
"""

import io
import numpy as np
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple

# Audio processing constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 4096  # 256ms at 16kHz

# Audio processing functions
async def process_audio_chunk(client_id: str, audio_chunk: bytes) -> None:
    """Process an audio chunk from a client."""
    session = session_store.get_session(client_id)
    if not session:
        logger.warning(f"No session found for client {client_id}")
        return
    
    # Add chunk to buffer
    session["audio_buffer"].extend(audio_chunk)
    
    # Check if we have enough data for VAD
    if len(session["audio_buffer"]) >= CHUNK_SIZE * 2:  # At least 512ms of audio
        # Detect voice activity
        is_speaking = await detect_voice_activity(client_id, bytes(session["audio_buffer"]))
        
        # Update session
        session_store.update_session(client_id, {"is_speaking": is_speaking})
        
        # If speaking, transcribe audio
        if is_speaking:
            # Keep accumulating audio for better transcription
            if len(session["audio_buffer"]) >= SAMPLE_RATE * 2:  # 2 seconds of audio
                # Transcribe audio
                transcription = await transcribe_audio(client_id, bytes(session["audio_buffer"]))
                
                # If we got a transcription, clear buffer
                if transcription:
                    session["audio_buffer"] = bytearray()
        else:
            # Not speaking, check if we have accumulated transcription
            if session["transcription_buffer"]:
                # Process transcription
                await process_transcription(client_id, session["transcription_buffer"])
                
                # Clear transcription buffer
                session_store.update_session(client_id, {"transcription_buffer": ""})
            
            # Clear audio buffer if not speaking
            session["audio_buffer"] = bytearray()

async def detect_voice_activity(client_id: str, audio_data: bytes) -> bool:
    """
    Detect voice activity in audio data using the VAD service.
    
    Args:
        client_id: Client ID
        audio_data: Audio data as bytes
        
    Returns:
        True if voice activity detected, False otherwise
    """
    try:
        # Prepare form data
        form_data = aiohttp.FormData()
        form_data.add_field("file", io.BytesIO(audio_data), filename="audio.raw")
        
        # Send request to VAD service
        async with http_session.post(f"{VAD_SERVICE_URL}/detect", data=form_data) as response:
            if response.status != 200:
                logger.warning(f"VAD service returned status {response.status}")
                return False
            
            # Parse response
            result = await response.json()
            
            # Check if speech segments were detected
            speech_segments = result.get("speech_segments", [])
            is_speaking = len(speech_segments) > 0
            
            # Log result
            if is_speaking:
                logger.debug(f"Voice activity detected for client {client_id}")
            
            return is_speaking
    except Exception as e:
        logger.error(f"Error detecting voice activity: {e}")
        return False

async def transcribe_audio(client_id: str, audio_data: bytes) -> Optional[str]:
    """
    Transcribe audio data using the STT service.
    
    Args:
        client_id: Client ID
        audio_data: Audio data as bytes
        
    Returns:
        Transcription text or None if failed
    """
    try:
        # Prepare form data
        form_data = aiohttp.FormData()
        form_data.add_field("file", io.BytesIO(audio_data), filename="audio.raw")
        form_data.add_field("language", "en")
        
        # Send request to STT service
        async with http_session.post(f"{STT_SERVICE_URL}/transcribe", data=form_data) as response:
            if response.status != 200:
                logger.warning(f"STT service returned status {response.status}")
                return None
            
            # Parse response
            result = await response.json()
            
            # Extract transcription
            text = result.get("text", "").strip()
            
            # If we got text, update session
            if text:
                session = session_store.get_session(client_id)
                if session:
                    # Append to transcription buffer
                    current_buffer = session.get("transcription_buffer", "")
                    if current_buffer:
                        current_buffer += " "
                    current_buffer += text
                    
                    # Update session
                    session_store.update_session(client_id, {"transcription_buffer": current_buffer})
                    
                    # Log transcription
                    logger.info(f"Transcription for client {client_id}: {text}")
                    
                    # Send transcription to client
                    await manager.send_json(client_id, {
                        "type": "transcription",
                        "text": text,
                        "is_final": result.get("is_final", False)
                    })
            
            return text
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return None

async def process_transcription(client_id: str, text: str) -> None:
    """
    Process a transcription from a client.
    
    Args:
        client_id: Client ID
        text: Transcription text
    """
    session = session_store.get_session(client_id)
    if not session:
        logger.warning(f"No session found for client {client_id}")
        return
    
    # Add user message to conversation
    session["conversation"].append({
        "role": "user",
        "content": text
    })
    
    # Generate response
    response = await generate_response(client_id, session["conversation"])
    
    # If we got a response, add it to conversation and synthesize speech
    if response:
        # Add assistant message to conversation
        session["conversation"].append({
            "role": "assistant",
            "content": response
        })
        
        # Synthesize speech
        await synthesize_speech(client_id, response)

# WebSocket endpoint for audio streaming
@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    """WebSocket endpoint for audio streaming."""
    # Generate client ID
    client_id = str(uuid.uuid4())
    
    # Accept connection
    await manager.connect(websocket, client_id)
    
    # Create session
    session = session_store.create_session(client_id)
    
    # Create task for sending TTS audio
    tts_task = asyncio.create_task(send_tts_audio(client_id))
    
    try:
        # Send client ID to client
        await websocket.send_json({"type": "client_id", "client_id": client_id})
        
        # Process incoming audio chunks
        while True:
            # Receive audio chunk
            audio_chunk = await websocket.receive_bytes()
            
            # Process audio chunk
            await process_audio_chunk(client_id, audio_chunk)
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
    finally:
        # Clean up
        manager.disconnect(client_id)
        session_store.delete_session(client_id)
        tts_task.cancel()
        try:
            await tts_task
        except asyncio.CancelledError:
            pass

async def send_tts_audio(client_id: str) -> None:
    """
    Send TTS audio to a client.
    
    Args:
        client_id: Client ID
    """
    session = session_store.get_session(client_id)
    if not session:
        logger.warning(f"No session found for client {client_id}")
        return
    
    try:
        while True:
            # Get audio from queue
            audio_data = await session["response_audio_queue"].get()
            
            # Send audio to client
            await manager.send_bytes(client_id, audio_data)
            
            # Mark task as done
            session["response_audio_queue"].task_done()
    except asyncio.CancelledError:
        logger.info(f"TTS audio task cancelled for client {client_id}")
    except Exception as e:
        logger.error(f"Error sending TTS audio: {e}")

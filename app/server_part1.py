#!/usr/bin/env python3
"""
Realtime Voice Chat Server - Part 1: Core and Configuration
"""

import os
import sys
import logging
import asyncio
import json
import time
import uuid
import dotenv
from typing import Dict, List, Optional, Any, Union
import aiohttp
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("voice-chat-server")

# Configuration
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")

# Service URLs
VAD_SERVICE_URL = os.getenv("VAD_SERVICE_URL", "http://vad:3001")
STT_SERVICE_URL = os.getenv("STT_SERVICE_URL", "http://stt:3002")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://tts:3003")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://llm:3004")
OLLAMA_SERVICE_URL = os.getenv("OLLAMA_SERVICE_URL", "http://ollama:11434")

# Create FastAPI app
app = FastAPI(title="Realtime Voice Chat Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTTP client session
http_session = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_text(self, client_id: str, message: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
    
    async def send_bytes(self, client_id: str, data: bytes):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_bytes(data)
    
    async def send_json(self, client_id: str, data: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(data)
    
    async def broadcast_text(self, message: str, exclude: Optional[str] = None):
        for client_id, connection in self.active_connections.items():
            if exclude is None or client_id != exclude:
                await connection.send_text(message)
    
    async def broadcast_json(self, data: dict, exclude: Optional[str] = None):
        for client_id, connection in self.active_connections.items():
            if exclude is None or client_id != exclude:
                await connection.send_json(data)

# Create connection manager
manager = ConnectionManager()

# Session data store
class SessionStore:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, client_id: str) -> Dict[str, Any]:
        """Create a new session for a client."""
        session = {
            "client_id": client_id,
            "created_at": time.time(),
            "conversation": [],
            "is_speaking": False,
            "last_activity": time.time(),
            "audio_buffer": bytearray(),
            "transcription_buffer": "",
            "response_audio_queue": asyncio.Queue(),
        }
        self.sessions[client_id] = session
        return session
    
    def get_session(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by client ID."""
        return self.sessions.get(client_id)
    
    def update_session(self, client_id: str, data: Dict[str, Any]) -> None:
        """Update a session with new data."""
        if client_id in self.sessions:
            self.sessions[client_id].update(data)
            self.sessions[client_id]["last_activity"] = time.time()
    
    def delete_session(self, client_id: str) -> None:
        """Delete a session."""
        if client_id in self.sessions:
            del self.sessions[client_id]
    
    def cleanup_old_sessions(self, max_age_seconds: int = 3600) -> None:
        """Clean up sessions older than max_age_seconds."""
        now = time.time()
        to_delete = []
        for client_id, session in self.sessions.items():
            if now - session["last_activity"] > max_age_seconds:
                to_delete.append(client_id)
        
        for client_id in to_delete:
            self.delete_session(client_id)
            logger.info(f"Cleaned up inactive session for client {client_id}")

# Create session store
session_store = SessionStore()

# Service health check
async def check_service_health(url: str) -> bool:
    """Check if a service is healthy."""
    try:
        async with http_session.get(f"{url}/healthz", timeout=2) as response:
            return response.status == 200
    except Exception as e:
        logger.warning(f"Health check failed for {url}: {e}")
        return False

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    global http_session
    
    # Create HTTP session
    http_session = aiohttp.ClientSession()
    
    # Check service health
    services = [
        ("VAD", VAD_SERVICE_URL),
        ("STT", STT_SERVICE_URL),
        ("TTS", TTS_SERVICE_URL),
        ("LLM", LLM_SERVICE_URL),
    ]
    
    for name, url in services:
        is_healthy = await check_service_health(url)
        logger.info(f"{name} service health: {'OK' if is_healthy else 'FAIL'}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    if http_session:
        await http_session.close()

# Health check endpoint
@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

# Home page
@app.get("/", response_class=HTMLResponse)
async def get_home():
    """Serve the home page."""
    with open("static/index.html", "r") as f:
        return f.read()

# API endpoints
class SystemInfoResponse(BaseModel):
    services: Dict[str, bool]
    active_connections: int
    uptime: float

@app.get("/api/system-info", response_model=SystemInfoResponse)
async def get_system_info():
    """Get system information."""
    # Check service health
    services = {
        "vad": await check_service_health(VAD_SERVICE_URL),
        "stt": await check_service_health(STT_SERVICE_URL),
        "tts": await check_service_health(TTS_SERVICE_URL),
        "llm": await check_service_health(LLM_SERVICE_URL),
    }
    
    return {
        "services": services,
        "active_connections": len(manager.active_connections),
        "uptime": time.time() - startup_time,
    }

# Store startup time
startup_time = time.time()

def main():
    """Run the server."""
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)

# Realtime Voice Chat Microservices

This project implements a modular, microservices-based architecture for a real-time voice chat system. Each AI component is decoupled into independent services that can be deployed separately, allowing for flexible scaling and resource allocation.

## Architecture Overview

The system is composed of the following microservices:

1. **VAD (Voice Activity Detection)** - Detects when a user is speaking
2. **STT (Speech-to-Text)** - Transcribes audio to text
3. **TTS (Text-to-Speech)** - Converts text to speech
4. **LLM (Language Model)** - Generates responses based on conversation context
5. **Main Application** - Orchestrates the flow between services and manages the user interface

```
        [ User Mic Input ]
               ↓
     ┌────────────────────┐
     │      Main App      │
     │  (stream manager)  │
     └────────────────────┘
     ↓        ↓         ↓         ↓
 [VAD API] [STT API] [TTS API] [LLM API]
     ↓        ↓         ↓         ↓
CPU低配   CPU部署    CPU部署    GPU部署
```

## Service Endpoints

Each service exposes a standardized API:

| Service | Endpoint | Request Format | Response Format |
|---------|----------|----------------|-----------------|
| VAD | POST /detect | Audio data | JSON with speech segments |
| STT | POST /transcribe | Audio data | JSON with transcribed text |
| TTS | POST /synthesize | JSON with text | Audio WAV or base64 |
| LLM | POST /generate | JSON with conversation | JSON with model response |

All services also provide:
- WebSocket endpoints for streaming data
- `/healthz` endpoint for health checks

## Deployment Architecture

This system is designed for a distributed deployment across multiple servers:

1. **CPU Server(s)**: Hosts the VAD, STT, TTS, and main application services
2. **GPU Server**: Hosts the Ollama service for LLM inference

### Deployment Steps

#### 1. GPU Server Setup (for Ollama)

Ollama is **not containerized** and should be installed directly on a GPU server:

```bash
# On the GPU server
# Install Ollama following instructions at https://ollama.ai/download
# For Linux:
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull the required model
ollama pull qwen3:14b
```

Make note of the GPU server's IP address, as you'll need to configure it in the `.env` file.

#### 2. CPU Server Setup (for microservices)

On your CPU server, deploy the remaining services using Docker Compose:

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file to configure your services
# IMPORTANT: Set OLLAMA_BASE_URL to point to your GPU server
# Example: OLLAMA_BASE_URL=http://192.168.1.100:11434
nano .env

# Start all services (except Ollama)
docker-compose up -d

# To view logs
docker-compose logs -f

# To stop all services
docker-compose down
```

### Individual Service Deployment

Each service can also be deployed independently:

```bash
# Example: Deploy only the STT service
cd stt
docker build -t voice-chat-stt .
docker run -p 3002:3002 -e STT_MODEL=openai/whisper-base voice-chat-stt
```

## Configuration

Each service can be configured through environment variables. See `.env.example` for all available options.

### Key Configuration Options

- **VAD Service**: Model selection, sensitivity parameters
- **STT Service**: Whisper model size, language settings
- **TTS Service**: Voice selection, audio quality settings
- **LLM Service**: Model selection (Ollama or OpenAI), API keys

## Development

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- NVIDIA GPU with CUDA support (optional, for faster inference)

### Local Development

For development without Docker:

```bash
# Example: Run the STT service locally
cd stt
pip install -r requirements.txt
python app.py
```

## API Examples

### VAD Service

```bash
# Detect voice activity in an audio file
curl -X POST http://localhost:3001/detect \
  -F "file=@audio_sample.wav"
```

### STT Service

```bash
# Transcribe audio
curl -X POST http://localhost:3002/transcribe \
  -F "file=@audio_sample.wav" \
  -F "language=en"
```

### TTS Service

```bash
# Generate speech from text
curl -X POST http://localhost:3003/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how can I help you today?", "voice": "en_female", "return_format": "wav"}'
```

### LLM Service

```bash
# Generate a response
curl -X POST http://localhost:3004/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me about microservices."}
    ],
    "temperature": 0.7
  }'
```

## Troubleshooting

- **Service not starting**: Check logs with `docker-compose logs [service_name]`
- **GPU not detected**: Ensure NVIDIA drivers and nvidia-docker are properly installed
- **High latency**: Consider using smaller models or upgrading hardware

## License

[MIT License](LICENSE)

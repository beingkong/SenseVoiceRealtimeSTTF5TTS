# Realtime Voice Chat Microservices

This project implements a modular, microservices-based architecture for a real-time voice chat system. Each AI component is decoupled into independent services that can be deployed separately, allowing for flexible scaling and resource allocation.

## Architecture Overview

The system is composed of the following microservices:

1. **VAD (Voice Activity Detection using SenseVoice)** - Detects when a user is speaking
2. **STT (Speech-to-Text using RealtimeSTT)** - Transcribes audio to text
3. **TTS (Text-to-Speech using F5 TTS)** - Converts text to speech
4. **LLM (Language Model using Qwen3 14b)** - Generates responses based on conversation context
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
SenseVoice RealtimeSTT  F5 TTS   Qwen3 14b
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

## Deployment Options

This system can be deployed in two ways:

1. **Docker Deployment**: Using Docker Compose for containerized deployment
2. **Non-Docker Deployment**: Running services directly on the host machine

### Docker Deployment

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

### Non-Docker Deployment

For deployment without Docker, use the provided `run_services.py` script:

#### Prerequisites

- Python 3.8 or higher
- Virtual environment module (`python -m venv`)
- For GPU acceleration: CUDA and appropriate drivers

#### Setup and Installation

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file to configure your services
# IMPORTANT: Set OLLAMA_BASE_URL to point to your GPU server
# Example: OLLAMA_BASE_URL=http://192.168.1.100:11434
nano .env

# Install dependencies for all services
python run_services.py --install

# List available services
python run_services.py --list
```

#### Running Services

```bash
# Start all services
python run_services.py --all

# Start a specific service
python run_services.py --service vad

# Stop a specific service
python run_services.py --stop vad

# Stop all services
python run_services.py --stop-all
```

The script will:
1. Create a virtual environment if it doesn't exist
2. Install all required dependencies
3. Start services in the correct order
4. Provide a web interface at http://localhost:8000

## Individual Service Deployment

Each service can also be deployed independently:

```bash
# Example: Deploy only the STT service with Docker
cd stt
docker build -t voice-chat-stt .
docker run -p 3002:3002 -e STT_MODEL=openai/whisper-base voice-chat-stt

# Example: Deploy only the STT service without Docker
cd stt
pip install -r requirements.txt
python app.py
```

## Configuration

Each service can be configured through environment variables. See `.env.example` for all available options.

### Key Configuration Options

- **VAD Service**: SenseVoice configuration, sensitivity parameters
- **STT Service**: RealtimeSTT settings, language settings
- **TTS Service**: F5 TTS voice selection, audio quality settings
- **LLM Service**: Qwen3 14b model settings, API configuration

## Development

### Prerequisites

- Docker and Docker Compose (for Docker deployment)
- Python 3.8+ (for non-Docker deployment)
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

- **Service not starting**: Check logs with `docker-compose logs [service_name]` or check the console output in non-Docker mode
- **GPU not detected**: Ensure NVIDIA drivers and CUDA are properly installed
- **High latency**: Consider using smaller models or upgrading hardware
- **Dependency issues**: For non-Docker deployment, try reinstalling dependencies with `python run_services.py --install`

## License

[MIT License](LICENSE)

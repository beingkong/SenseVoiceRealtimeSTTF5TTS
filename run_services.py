#!/usr/bin/env python3
"""
Non-Docker deployment script for Realtime Voice Chat services.
This script allows running all services without Docker.
"""

import os
import sys
import argparse
import subprocess
import time
import signal
import atexit
import platform
import shutil
import socket
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
log_dir = Path("logs")
if not log_dir.exists():
    log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"run_services_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("run_services")

# Define service information
SERVICES = {
    "vad": {
        "dir": "vad",
        "port": 3001,
        "description": "Voice Activity Detection (SenseVoice)",
        "dependencies": ["numpy", "torch", "soundfile"]
    },
    "stt": {
        "dir": "stt",
        "port": 3002,
        "description": "Speech-to-Text (RealtimeSTT)",
        "dependencies": ["numpy", "torch", "transformers"]
    },
    "tts": {
        "dir": "tts",
        "port": 3003,
        "description": "Text-to-Speech (F5 TTS)",
        "dependencies": ["numpy", "torch", "gTTS"]
    },
    "llm": {
        "dir": "llm",
        "port": 3004,
        "description": "Language Model (Qwen3 14b)",
        "dependencies": ["httpx", "openai"]
    },
    "app": {
        "dir": "app",
        "port": 8000,
        "description": "Main Application",
        "dependencies": ["aiohttp", "websockets", "python-dotenv"]
    }
}

# Global variables
processes = {}
python_executable = sys.executable
is_windows = platform.system() == "Windows"

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)

def find_venv():
    """Use system Python directly."""
    logger.info("Using system Python directly")
    return python_executable

def install_dependencies(venv_python, service=None):
    """Install dependencies for all services or a specific service."""
    if service:
        if service not in SERVICES:
            logger.error(f"Unknown service '{service}'")
            return False
        
        services_to_install = [service]
    else:
        services_to_install = SERVICES.keys()
    
    # First, upgrade pip
    logger.info("Upgrading pip...")
    try:
        subprocess.run(
            [venv_python, "-m", "pip", "install", "--upgrade", "pip"],
            check=True
        )
        logger.info("Pip upgraded successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error upgrading pip: {e}")
        # Continue anyway
    
    # Install common dependencies needed by all services
    logger.info("Installing common dependencies...")
    try:
        subprocess.run(
            [venv_python, "-m", "pip", "install", "fastapi", "uvicorn", "websockets", "python-multipart"],
            check=True
        )
        logger.info("Common dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing common dependencies: {e}")
        logger.info("Trying with system Python...")
        try:
            subprocess.run(
                [python_executable, "-m", "pip", "install", "fastapi", "uvicorn", "websockets", "python-multipart"],
                check=True
            )
            logger.info("Common dependencies installed successfully with system Python.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing common dependencies with system Python: {e}")
            # Continue anyway
    
    for svc in services_to_install:
        req_file = Path(SERVICES[svc]["dir"]) / "requirements.txt"
        if req_file.exists():
            logger.info(f"Installing dependencies for {svc} service...")
            try:
                result = subprocess.run(
                    [venv_python, "-m", "pip", "install", "-r", str(req_file)],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"Error installing dependencies for {svc} service:")
                    logger.error(result.stderr)
                    
                    # Try to install core dependencies directly
                    core_deps = SERVICES[svc]["dependencies"]
                    logger.info(f"Trying to install core dependencies: {', '.join(core_deps)}")
                    try:
                        subprocess.run(
                            [venv_python, "-m", "pip", "install"] + core_deps,
                            check=True
                        )
                        logger.info(f"Core dependencies for {svc} service installed successfully.")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"Error installing core dependencies: {e}")
                        # Continue anyway
                else:
                    logger.info(f"Dependencies for {svc} service installed successfully.")
            except Exception as e:
                logger.error(f"Error during dependency installation: {e}")
                # Continue anyway
        else:
            logger.warning(f"requirements.txt not found for {svc} service at {req_file}")
            
            # Try to install core dependencies directly
            core_deps = SERVICES[svc]["dependencies"]
            logger.info(f"Trying to install core dependencies: {', '.join(core_deps)}")
            try:
                subprocess.run(
                    [venv_python, "-m", "pip", "install"] + core_deps,
                    check=True
                )
                logger.info(f"Core dependencies for {svc} service installed successfully.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error installing core dependencies: {e}")
                # Continue anyway
    
    return True

def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("Creating .env file from .env.example...")
        shutil.copy(env_example, env_file)
        
        # Update service URLs for non-Docker deployment
        with open(env_file, "r") as f:
            content = f.read()
        
        # Replace Docker service URLs with localhost
        content = content.replace("http://vad:3001", "http://localhost:3001")
        content = content.replace("http://stt:3002", "http://localhost:3002")
        content = content.replace("http://tts:3003", "http://localhost:3003")
        content = content.replace("http://llm:3004", "http://localhost:3004")
        
        with open(env_file, "w") as f:
            f.write(content)
        
        print(".env file created and updated for non-Docker deployment.")
    elif not env_example.exists():
        print("Warning: .env.example not found, cannot create .env file.")

def is_port_in_use(port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_service(venv_python, service, wait=True):
    """Start a specific service."""
    if service not in SERVICES:
        logger.error(f"Unknown service '{service}'")
        return False
    
    service_info = SERVICES[service]
    service_dir = service_info["dir"]
    port = service_info["port"]
    app_file = Path(service_dir) / "app.py"
    
    # Check if port is already in use
    if is_port_in_use(port):
        logger.warning(f"Port {port} is already in use. Service {service} may not start correctly.")
    
    # Print debug information
    print(f"Service directory: {service_dir}")
    print(f"App file path: {app_file}")
    print(f"App file exists: {app_file.exists()}")
    
    # Use absolute path to ensure we're accessing the correct file
    abs_app_file = Path.cwd() / service_dir / "app.py"
    print(f"Absolute app file path: {abs_app_file}")
    print(f"Absolute app file exists: {abs_app_file.exists()}")
    
    # Use the absolute path
    app_file = abs_app_file
    
    # Special handling for the app service
    if service == "app":
        # For the app service, we'll use combine_server_parts.py directly
        combine_script = Path(service_dir) / "combine_server_parts.py"
        abs_combine_script = Path.cwd() / service_dir / "combine_server_parts.py"
        
        logger.info(f"Combine script path: {combine_script}")
        logger.info(f"Combine script exists: {combine_script.exists()}")
        logger.info(f"Absolute combine script path: {abs_combine_script}")
        logger.info(f"Absolute combine script exists: {abs_combine_script.exists()}")
        
        # Use the absolute path
        combine_script = abs_combine_script
        
        if combine_script.exists():
            logger.info("Combining server parts for main application...")
            try:
                # Try to run the combine script with the virtual environment Python
                try:
                    subprocess.run(
                        [venv_python, str(combine_script)],
                        cwd=service_dir,
                        check=True
                    )
                except FileNotFoundError:
                    # If virtual environment Python is not found, try with system Python
                    logger.warning(f"Python executable not found: {venv_python}")
                    logger.info("Trying with system Python instead...")
                    subprocess.run(
                        [python_executable, str(combine_script)],
                        cwd=service_dir,
                        check=True
                    )
                
                # Check if app.py was created
                app_py_path = Path(service_dir) / "app.py"
                abs_app_py_path = Path.cwd() / service_dir / "app.py"
                
                if app_py_path.exists() or abs_app_py_path.exists():
                    logger.info("Combined app.py created successfully")
                    app_file = abs_app_py_path
                else:
                    # If app.py wasn't created, use the static/index.html file directly
                    logger.warning("app.py not created, using static/index.html directly")
                    static_dir = Path(service_dir) / "static"
                    if static_dir.exists():
                        # Start a simple HTTP server for the static files
                        logger.info("Starting HTTP server for static files...")
                        cmd = [venv_python, "-m", "http.server", str(SERVICES[service]["port"]), "--directory", str(static_dir)]
                        try:
                            if is_windows:
                                process = subprocess.Popen(
                                    cmd,
                                    creationflags=subprocess.CREATE_NEW_CONSOLE
                                )
                            else:
                                process = subprocess.Popen(cmd)
                            
                            processes[service] = process
                            logger.info(f"HTTP server started on port {SERVICES[service]['port']}")
                            return True
                        except Exception as e:
                            logger.error(f"Error starting HTTP server: {e}")
                            return False
                    else:
                        logger.error("Static directory not found")
                        return False
            except subprocess.CalledProcessError as e:
                logger.error(f"Error combining server parts: {e}")
                # Try to start a simple HTTP server for the static files
                static_dir = Path(service_dir) / "static"
                if static_dir.exists():
                    logger.info("Starting HTTP server for static files...")
                    cmd = [venv_python, "-m", "http.server", str(SERVICES[service]["port"]), "--directory", str(static_dir)]
                    try:
                        if is_windows:
                            process = subprocess.Popen(
                                cmd,
                                creationflags=subprocess.CREATE_NEW_CONSOLE
                            )
                        else:
                            process = subprocess.Popen(cmd)
                        
                        processes[service] = process
                        logger.info(f"HTTP server started on port {SERVICES[service]['port']}")
                        return True
                    except Exception as e2:
                        logger.error(f"Error starting HTTP server: {e2}")
                        return False
                else:
                    logger.error("Static directory not found")
                    return False
        else:
            logger.error(f"Error: combine_server_parts.py not found for app service")
            # Try to start a simple HTTP server for the static files
            static_dir = Path(service_dir) / "static"
            if static_dir.exists():
                logger.info("Starting HTTP server for static files...")
                cmd = [venv_python, "-m", "http.server", str(SERVICES[service]["port"]), "--directory", str(static_dir)]
                try:
                    if is_windows:
                        process = subprocess.Popen(
                            cmd,
                            creationflags=subprocess.CREATE_NEW_CONSOLE
                        )
                    else:
                        process = subprocess.Popen(cmd)
                    
                    processes[service] = process
                    logger.info(f"HTTP server started on port {SERVICES[service]['port']}")
                    return True
                except Exception as e:
                    logger.error(f"Error starting HTTP server: {e}")
                    return False
            else:
                logger.error("Static directory not found")
                return False
    elif not app_file.exists():
        print(f"Error: app.py not found for {service} service at {app_file}")
        return False
    
    print(f"Starting {service} service ({service_info['description']}) on port {service_info['port']}...")
    
    # Start the service
    cmd = [venv_python, str(app_file)]
    
    try:
        if is_windows:
            process = subprocess.Popen(
                cmd,
                cwd=service_dir,
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            process = subprocess.Popen(
                cmd,
                cwd=service_dir
            )
        
        processes[service] = process
        
        if wait:
            # Wait for service to start
            print(f"Waiting for {service} service to start...")
            time.sleep(3)
        
        return True
    except FileNotFoundError as e:
        print(f"Error starting {service} service: {e}")
        print(f"Python executable not found: {venv_python}")
        print("Trying with system Python instead...")
        
        # Try with system Python
        cmd = [python_executable, str(app_file)]
        try:
            if is_windows:
                process = subprocess.Popen(
                    cmd,
                    cwd=service_dir,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            else:
                process = subprocess.Popen(
                    cmd,
                    cwd=service_dir
                )
            
            processes[service] = process
            
            if wait:
                # Wait for service to start
                print(f"Waiting for {service} service to start...")
                time.sleep(3)
            
            return True
        except Exception as e2:
            print(f"Error starting {service} service with system Python: {e2}")
            return False
    except Exception as e:
        print(f"Error starting {service} service: {e}")
        return False

def stop_service(service):
    """Stop a specific service."""
    if service not in processes:
        print(f"Warning: {service} service is not running")
        return
    
    print(f"Stopping {service} service...")
    
    try:
        if is_windows:
            # On Windows, we need to use taskkill to kill the process tree
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(processes[service].pid)])
        else:
            # On Unix, we can use process.terminate()
            processes[service].terminate()
        
        # Wait for process to terminate
        try:
            processes[service].wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"Warning: {service} service did not terminate gracefully, forcing...")
            processes[service].kill()
        
        del processes[service]
    except Exception as e:
        print(f"Error stopping {service} service: {e}")

def stop_all_services():
    """Stop all running services."""
    for service in list(processes.keys()):
        stop_service(service)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run Realtime Voice Chat services without Docker")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    parser.add_argument("--service", type=str, help="Start a specific service")
    parser.add_argument("--all", action="store_true", help="Start all services")
    parser.add_argument("--stop", type=str, help="Stop a specific service")
    parser.add_argument("--stop-all", action="store_true", help="Stop all services")
    parser.add_argument("--list", action="store_true", help="List available services")
    
    args = parser.parse_args()
    
    # Check Python version
    check_python_version()
    
    # Find or create virtual environment
    venv_python = find_venv()
    
    # Register cleanup function
    atexit.register(stop_all_services)
    
    # Handle signals
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))
    
    if args.list:
        # List available services
        print("Available services:")
        for name, info in SERVICES.items():
            print(f"  {name}: {info['description']} (port {info['port']})")
        return
    
    if args.install:
        # Install dependencies
        create_env_file()
        if not install_dependencies(venv_python, args.service):
            sys.exit(1)
    
    if args.stop:
        # Stop a specific service
        stop_service(args.stop)
    
    if args.stop_all:
        # Stop all services
        stop_all_services()
    
    if args.service:
        # Start a specific service
        if not start_service(venv_python, args.service):
            sys.exit(1)
    
    if args.all:
        # Start all services
        logger.info("Starting all services...")
        
        # Create .env file if needed
        create_env_file()
        
        # Install dependencies for all services if not explicitly done
        if not args.install:
            logger.info("Installing dependencies for all services...")
            if not install_dependencies(venv_python):
                logger.error("Error installing dependencies. Please run with --install first.")
                sys.exit(1)
        
        # Install common dependencies needed by all services
        logger.info("Installing common dependencies...")
        try:
            subprocess.run(
                [venv_python, "-m", "pip", "install", "fastapi", "uvicorn", "websockets", "python-multipart"],
                check=True
            )
            logger.info("Common dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing common dependencies: {e}")
            logger.info("Trying with system Python...")
            try:
                subprocess.run(
                    [python_executable, "-m", "pip", "install", "fastapi", "uvicorn", "websockets", "python-multipart"],
                    check=True
                )
                logger.info("Common dependencies installed successfully with system Python.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error installing common dependencies with system Python: {e}")
                sys.exit(1)
        
        # Start services in order
        service_order = ["vad", "stt", "tts", "llm", "app"]
        
        for service in service_order:
            if not start_service(venv_python, service, wait=(service != "app")):
                print(f"Error starting {service} service, stopping all services...")
                stop_all_services()
                sys.exit(1)
        
        print("\nAll services started successfully!")
        print("Main application is running at: http://localhost:8000")
        print("\nPress Ctrl+C to stop all services.")
        
        # Keep the script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping all services...")
            stop_all_services()
    
    # If no action specified, show help
    if not (args.install or args.service or args.all or args.stop or args.stop_all or args.list):
        parser.print_help()

if __name__ == "__main__":
    main()

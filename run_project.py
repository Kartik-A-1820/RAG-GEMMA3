import os
import subprocess
import sys
import venv
import time
import socket
from pathlib import Path

# --- Config ---
project_root = Path(__file__).parent.resolve()
venv_dir = project_root / "venv"
requirements = project_root / "requirements.txt"
frontend_file = project_root / "Frontend" / "app.py"
backend_module = "backend.main_4bit:app"  # exact uvicorn format

# --- 1. Ensure Python 3.11 ---
if not sys.version.startswith("3.11"):
    print("❌ Python 3.11 is required. You're using:", sys.version)
    sys.exit(1)

# --- 2. Determine venv paths ---
if os.name == "nt":
    pip_path = venv_dir / "Scripts" / "pip"
    python_path = venv_dir / "Scripts" / "python"
else:
    pip_path = venv_dir / "bin" / "pip"
    python_path = venv_dir / "bin" / "python"

# --- 3. Create venv and install dependencies ---
if not venv_dir.exists():
    print("📦 Creating virtual environment...")
    venv.create(venv_dir, with_pip=True)

    print("📚 Installing dependencies...")
    subprocess.run([str(pip_path), "install", "-r", str(requirements)])
else:
    print("✅ venv already exists. Skipping installation.")

# --- 4. Launch FastAPI backend with uvicorn ---
print("🚀 Launching FastAPI backend on 0.0.0.0:8000 with --reload...")
backend_proc = subprocess.Popen([
    str(python_path), "-m", "uvicorn", backend_module,
    "--host", "0.0.0.0", "--port", "8000", "--reload"
])

# --- 5. Wait until backend is ready ---
print("⏳ Waiting for backend to become ready...")

def is_port_open(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0

for _ in range(20):
    if is_port_open("localhost", 8000):
        print("✅ Backend is live at http://localhost:8000")
        break
    time.sleep(0.5)
else:
    print("❌ Backend didn't start in time.")
    backend_proc.terminate()
    sys.exit(1)

# --- 6. Launch Streamlit frontend ---
print("🎯 Launching Streamlit frontend...")
frontend_proc = subprocess.Popen([
    str(python_path), "-m", "streamlit", "run", str(frontend_file)
])

# --- 7. Wait for both to exit (optional)
backend_proc.wait()
frontend_proc.wait()

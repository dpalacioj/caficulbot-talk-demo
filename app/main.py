import subprocess
import webbrowser
import os
import sys
import threading
import time

def start_services():
    script = "run-local.sh"
    subprocess.Popen(["bash", script])

def open_web():
    time.sleep(20)
    webbrowser.open("http://localhost:8501")

if __name__ == "__main__":
    threading.Thread(target=start_services, daemon=True).start()
    open_web()
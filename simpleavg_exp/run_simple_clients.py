import os
import subprocess

PYTHON = r"C:\Users\SHIREEN\OneDrive\Desktop\MAJOR PROJECT\FederatedLearning_Project\venv\Scripts\python.exe"

CLIENTS = ["ElBorn", "LesCorts", "PobleSec"]

for city in CLIENTS:
    env = os.environ.copy()
    env["SERVER_ADDRESS"] = "localhost:8090"

    subprocess.Popen([PYTHON, "clients/fl_client.py", city], env=env)

print("🚀 SimpleAvg clients started on port 8090")

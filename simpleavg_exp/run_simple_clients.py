# simpleavg_exp/run_simple_clients.py

import os
import subprocess
import sys

# ✅ Current active Python interpreter use karo
PYTHON = sys.executable

# ✅ Clients list
CLIENTS = ["ElBorn", "LesCorts", "PobleSec"]

for city in CLIENTS:
    env = os.environ.copy()
    env["SERVER_ADDRESS"] = "localhost:8090"

    # clients/fl_client.py path project root se relative hai
    subprocess.Popen([PYTHON, "clients/fl_client.py", city], env=env)

print("🚀 SimpleAvg clients started on port 8090")

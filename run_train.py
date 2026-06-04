import subprocess
import sys
import os

os.environ["PYTHONPATH"] = "src"
log = open("train_output.log", "w", buffering=1)
err = open("train_err.log", "w", buffering=1)

proc = subprocess.Popen(
    [sys.executable, "-m", "models.train"],
    stdout=log, stderr=err,
    cwd=r"C:\Users\PC\Downloads\PTDLL",
    env={**os.environ, "PYTHONPATH": "src"},
)
with open("train_pid.txt", "w") as f:
    f.write(str(proc.pid))
print(f"Training started, PID: {proc.pid}", flush=True)

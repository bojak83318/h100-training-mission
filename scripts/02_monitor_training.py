import time
import re
import subprocess
from pathlib import Path

LOG_FILE = "outputs/glm4.7-rust-hybrid-fp4-v3/train.log"
KILL_FLAG = "outputs/glm4.7-rust-hybrid-fp4-v3/KILL_TRAINING"

def kill_training(reason):
    print(f"❌ KILL TRIGGERED: {reason}")
    Path(KILL_FLAG).touch()
    subprocess.run(["pkill", "-f", "llamafactory-cli"])

def monitor():
    print("🔍 Monitor Active. Watching for anomalies...")
    while not Path(KILL_FLAG).exists():
        time.sleep(10)
        if not Path(LOG_FILE).exists(): continue
        
        # Tail log
        try:
            with open(LOG_FILE) as f: lines = f.readlines()[-20:]
        except: continue

        for line in lines:
            # Extract metrics
            loss_match = re.search(r"'loss':\s*([\d.]+)", line)
            step_match = re.search(r"'step':\s*(\d+)", line)
            
            if loss_match and step_match:
                loss = float(loss_match.group(1))
                step = int(step_match.group(1))
                
                # 1. Loss Explosion
                if loss > 8.0: kill_training(f"Loss Explosion ({loss}) at step {step}")
                
                # 2. Early Stall (The Cliff Check)
                if step == 50 and loss > 4.0: kill_training(f"Loss Cliff Missed ({loss} > 4.0) at Step 50")
                
                # 3. Convergence Failure
                if step == 200 and loss > 3.0: kill_training(f"Convergence Failure ({loss} > 3.0) at Step 200")
                
if __name__ == "__main__":
    monitor()

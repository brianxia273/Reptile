# Runner script for full training pipeline

import subprocess
import config
import time

startTime = time.time()
print(f"Starting Full Run; DO NOT ADJUST CONFIG WHILE RUNNING. Size {config.p2Size}")
subprocess.run(["python", "All-DataGenerate.py"])
subprocess.run(["python", "All-PreTrain.py"])
subprocess.run(["python", "All-MetaTrain.py"])
subprocess.run(["python", "All-FineTune.py"])
print(f"Time Elapsed: {time.time() - startTime}")
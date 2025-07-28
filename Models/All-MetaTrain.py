# Runner script for FCNN and 1D-Conv MetaTrain scripts
import subprocess
import config

print(f"Starting MetaTrain; DO NOT ADJUST CONFIG P3 WHILE RUNNING. Size {config.p2Size}")
print("Starting FCNN MetaTrain")
subprocess.run(["python", "FCNN-MetaTrain.py"])
print("Starting 1D-Conv MetaTrain")
subprocess.run(["python", "1D-Conv-MetaTrain.py"])
print("Finished all MetaTrain!")

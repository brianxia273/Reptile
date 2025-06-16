# Runner script for DataGenerate scripts of SVR, BRR, and GPR regression models
import subprocess

print("Starting all DataGenerate")
subprocess.run(["python", "SVR-DataGenerate-MetaTrain.py"])
subprocess.run(["python", "SVR-DataGenerate-PreTrain.py"])
subprocess.run(["python", "BRR-DataGenerate.py"])
subprocess.run(["python", "GPR-DataGenerate.py"])
print("Completed all DataGenerate!")

# Runner script for DataGenerate scripts of SVR, BRR, and GPR regression models
import subprocess

print("Starting all DataGenerate; DO NOT ADJUST CONFIG P1 WHILE RUNNING")
subprocess.run(["python", "SVR-DataGenerate-MetaTrain.py"])
subprocess.run(["python", "SVR-DataGenerate-PreTrain.py"])
subprocess.run(["python", "BRR-DataGenerate.py"])
subprocess.run(["python", "GPR-DataGenerate.py"])
print("Completed all DataGenerate!")

# Testing Fine-Tuned Neural Network Model's accuracy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pandas as pd
import config
import os
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt

# Select from config
data = os.path.join("Datasets" , config.p5Data)
yIndex = config.p5YIndex
nn = config.p5NN
setSize = config.p5NNSize
nnEpoch = config.p5NNEpoch
nnBatch = config.p5NNBatch
augmentedDataCount = config.p5N

datasetModels  = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
output = "Film Thickness" if yIndex == -2 else "NTi"

# Import NN
mlModelPath = os.path.join("Fine-Tuned Neural Networks", nn, datasetModels, output, f"Fine-Tuned {nn} - N_{augmentedDataCount} Size_{setSize} Epoch_{nnEpoch} Batch_{nnBatch}.keras")
mlModel = load_model(mlModelPath)

# Import Real Data CSV file
df = pd.read_csv(data)
x = df.iloc[:, :-2].values
y = df.iloc[:, yIndex].values   # Selecting output

# Normalize data
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xLog = np.log1p(x)
xScaled = dataScaler.fit_transform(xLog)

# Check predictions
yPredict = mlModel.predict(xScaled)
mse = mean_squared_error(y, yPredict)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y - yPredict) / y))
ev = explained_variance_score(y, yPredict)
r2 = r2_score(y, yPredict)

# Print accuracy
print(f"{mlModelPath} Model Evaluation Metrics: ")
print(f"MSE: {mse}\n")
print(f"RMSE: {rmse}\n")
print(f"MAPE: {mape}\n")
print(f"EV: {ev}\n")
print(f"R^2: {r2}\n")

plt.figure(figsize=(8, 6))
plt.scatter(y, yPredict, alpha=0.6, edgecolor='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # ideal line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.grid(True)
plt.tight_layout()
plt.show()
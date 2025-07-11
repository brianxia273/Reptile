# Testing Fine-Tuned Neural Network Model's accuracy
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import config
import os
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Select from config
data = os.path.join("Datasets", config.p5Data)
yIndex = config.p5YIndex
nn = config.p5NN
setSize = config.p5NNSize
nnEpoch = config.p5NNEpoch
nnBatch = config.p5NNBatch
augmentedDataCount = config.p5N
randomState = config.p5RandomState

datasetModels = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
output = "Film Thickness" if yIndex == -2 else "NTi"

# Import NN
mlModelPath = os.path.join("Fine-Tuned Neural Networks", nn, datasetModels, output,
                           f"Fine-Tuned {nn} - N_{augmentedDataCount} Size_{setSize} Epoch_{nnEpoch} Batch_{nnBatch}.keras")
mlModel = load_model(mlModelPath)

# Import Real Data CSV file
df = pd.read_csv(data)
x = df.iloc[:, :-2].values
y = df.iloc[:, yIndex].values  # Selecting output

# Train/Test split, use Test for evaluation
trainSize = min(setSize, int(0.8 * len(x)), len(x))
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=randomState)

# Normalize data, ignore Train
scalerName = f"Fine-Tuned {nn} - N_{augmentedDataCount} Size_{setSize} Epoch_{nnEpoch} Batch_{nnBatch} DataScaler.pkl"
dataScaler = joblib.load(os.path.join("Data Scalers", nn, datasetModels, output, scalerName))
xTestLog= np.log1p(xTest)
xTestScaled = dataScaler.transform(xTestLog)

# Check predictions
yPredict = mlModel.predict(xTestScaled)
mse = mean_squared_error(yTest, yPredict)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(yTest, yPredict)
ev = explained_variance_score(yTest, yPredict)
r2 = r2_score(yTest, yPredict)

# Print accuracy
print(f"{mlModelPath} Model Evaluation Metrics: ")
print(f"Seed: {randomState}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}")
print(f"EV: {ev}")
print(f"R^2: {r2}")

plt.figure(figsize=(8, 6))
plt.scatter(yTest, yPredict, alpha=0.6, edgecolor='k')
plt.plot([yTest.min(), yTest.max()], [yTest.min(), yTest.max()], 'r--', lw=2)  # ideal line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.grid(True)
plt.tight_layout()
plt.show()

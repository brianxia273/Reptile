# Testing Fine-Tuned Neural Network Model's accuracy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import config
import os


# Select from config
nn = config.taNN
data = os.path.join("Datasets" , config.taData)
yIndex = config.taYIndex
setSize = config.taNNSize
nnEpoch = config.taNNEpochs
nnBatch = config.taNNBatch

datasetModels  = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
output = "Film Thickness" if yIndex == -2 else "NTi"

# Import NN
mlModelPath = os.path.join("Fine-Tuned Neural Networks", nn, datasetModels, output, f"Fine-Tuned {nn} - Size_{setSize} Epoch_{nnEpoch} Batch_{nnBatch}.keras")
mlModel = load_model(mlModelPath)

# Import Real Data CSV file
df = pd.read_csv(data)
x = df.iloc[:, :-2].values
y = df.iloc[:, yIndex].values   # Selecting output

# Normalize data
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xLog = np.log1p(x)
xScaled = dataScaler.fit_transform(xLog)



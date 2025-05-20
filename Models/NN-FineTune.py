# Fine-tuning Neural Network Model after Meta-Learning

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import config
import os
import joblib

# Select from config
nn = config.p4NN
yIndex = config.p4YIndex
data = os.path.join("Datasets", config.p4Data)
nnEpochs = config.p4NNEpoch
nnBatch = config.p4NNBatch
epochs = config.p4Epochs
batchSize = config.p4BatchSize
setSize = config.p4NNSize
learningRate = config.p4LearningRate
augmentedDataCount = config.p4N

datasetModels = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
output = "Film Thickness" if yIndex == -2 else "NTi"

# Import NN
mlModelPath = os.path.join("Meta-Trained Neural Networks", nn, datasetModels, output,
                           f"Meta-Trained {nn} - N_{augmentedDataCount} Size_{setSize} Epoch_{nnEpochs} Batch_{nnBatch}.keras")
mlModel = load_model(mlModelPath)
mlModel.compile(optimizer=Adam(learning_rate=learningRate), loss='mse')

# Import Real Data CSV file
df = pd.read_csv(data)
x = df.iloc[:, :-2].values
y = df.iloc[:, yIndex].values  # Selecting output

# Normalize data
mlScalerPath = os.path.join("Data Scalers", nn, datasetModels, output,
                            f"Meta-Trained {nn} - N_{augmentedDataCount} Size_{setSize} Epoch_{nnEpochs} Batch_{nnBatch} DataScaler.pkl")
dataScaler = joblib.load(mlScalerPath)
xLog = np.log1p(x)
xScaled = dataScaler.transform(xLog)

# Fine-Tune NN
history = mlModel.fit(xScaled, y, epochs=epochs, batch_size=batchSize, verbose=1)  # Add learning rate

# Print history
print("Training Loss:", history.history['loss'])

# Save Fine-Tuned NN
directory = os.path.join("Fine-Tuned Neural Networks", nn, datasetModels, output)
os.makedirs(directory, exist_ok=True)
modelName = f"Fine-Tuned {nn} - N_{augmentedDataCount} Size_{setSize} Epoch_{epochs} Batch_{batchSize}.keras"
mlModel.save(os.path.join(directory, modelName))
print("Saved " + os.path.join(directory, modelName) + "!")

# Save Data Scaler
scalerDirectory = os.path.join("Data Scalers", nn, datasetModels, output)
os.makedirs(scalerDirectory, exist_ok=True)
scalerName = f"Fine-Tuned {nn} - N_{augmentedDataCount} Size_{setSize} Epoch_{epochs} Batch_{batchSize} DataScaler.pkl"
joblib.dump(dataScaler, os.path.join(scalerDirectory, scalerName))
print("Saved " + os.path.join(scalerDirectory, scalerName) + "!")

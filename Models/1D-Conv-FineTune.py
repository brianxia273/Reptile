# Fine-tuning Neural Network Model after Meta-Learning

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import config
import os
import joblib


# Select from config
nn = "1D-Conv"
yIndex = config.p4YIndex
data = os.path.join("Datasets", config.p4Data)
nnEpochs = config.p4NNEpoch
nnBatch = config.p4NNBatch
epochs = config.p4Epochs
batchSize = config.p4BatchSize
setSize = config.p4NNSize
learningRate = config.p4LearningRate
augmentedDataCount = config.p4N
randomState = config.p4RandomState

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

# Train/Test split
trainSize = min(setSize, int(0.8 * len(x)), len(x))
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=randomState)

# Normalize data
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xTrainLog = np.log1p(xTrain)
xTrainScaled = dataScaler.fit_transform(xTrainLog)
xTestLog = np.log1p(xTest)
xTestScaled = dataScaler.transform(xTestLog)

# Save Data Scaler
scalerDirectory = os.path.join("Data Scalers", nn, datasetModels, output)
os.makedirs(scalerDirectory, exist_ok=True)
scalerName = f"Fine-Tuned {nn} - N_{augmentedDataCount} Size_{setSize} Epoch_{epochs} Batch_{batchSize} DataScaler.pkl"
joblib.dump(dataScaler, os.path.join(scalerDirectory, scalerName))
print("Saved " + os.path.join(scalerDirectory, scalerName) + "!")

# Fine-Tune NN
history = mlModel.fit(xTrainScaled, yTrain, epochs=epochs, batch_size=batchSize, verbose=1, validation_data=(xTestScaled, yTest))

# Print history
# print("Training Loss:", history.history['loss'])

# Save Fine-Tuned NN
directory = os.path.join("Fine-Tuned Neural Networks", nn, datasetModels, output)
os.makedirs(directory, exist_ok=True)
modelName = f"Fine-Tuned {nn} - N_{augmentedDataCount} Size_{setSize} Epoch_{epochs} Batch_{batchSize}.keras"
mlModel.save(os.path.join(directory, modelName))
print("Saved " + os.path.join(directory, modelName) + "!")

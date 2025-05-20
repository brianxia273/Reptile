# Pre-training 1-dimensional convolution neura network using SVG InterExtra augmented data
# Pre-trained on SVR interpolated and extrapolated data

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
import pandas as pd
import config
import os

# Select size, dataset, output, randomState, data, epochs, and batch size from config
setSize = config.p2Size
yIndex = config.p2YIndex
randomState = config.p2RandomState
data = os.path.join("Datasets", config.p2Data)
model = "SVR"
epochs = config.p2Epochs
batchSize = config.p2BatchSize
learningRate = config.p2LearningRate
augmentedDataCount = config.p2N


def constructModelLayers():
    model = Sequential()

    # Layer 1
    model.add(Conv1D(filters=2, kernel_size=3, padding='same',
                     activation='relu',
                     kernel_initializer='random_normal',
                     kernel_regularizer=l1_l2(l1=1e-3, l2=1e-2),
                     input_shape=(8, 1)))
    model.add(BatchNormalization())

    # Layer 2
    model.add(Conv1D(filters=1, kernel_size=3, padding='same',
                     activation='relu',
                     kernel_initializer='random_normal',
                     kernel_regularizer=l1_l2(l1=1e-3, l2=1e-2)))
    model.add(BatchNormalization())

    # Flattening output
    model.add(Flatten())
    model.add(Dense(1, activation='linear',
                    kernel_initializer='random_normal',
                    kernel_regularizer=l1_l2(l1=1e-3, l2=1e-2)))
    model.compile(optimizer=Adam(learning_rate=learningRate), loss='mean_squared_error')
    return model


datasetModels = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
output = "Film Thickness" if yIndex == -2 else "NTi"

# Import Augmented Data CSV file
augDataDirectory = os.path.join("Regression Model Data and Metrics", datasetModels, output, model,
                                f"{model} PreTrain N_{augmentedDataCount} Size_{setSize} Random_{randomState} Augmented Data.csv")
augmentedData = pd.read_csv(augDataDirectory)
x = augmentedData.iloc[:, :-1].values
y = augmentedData.iloc[:, -1].values

# Normalize data
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xLog = np.log1p(x)
xScaled = dataScaler.fit_transform(xLog)

# Construct 1D-Conv Model
convModel = constructModelLayers()

# Pre-train 1D-Conv
history = convModel.fit(xScaled, y, epochs=epochs, batch_size=batchSize, verbose=1)

# Print history
# print("Training Loss:", history.history['loss'])

# Save 1D-Conv
directory = os.path.join("Pre-Trained Neural Networks", "1D-Conv", datasetModels, output)
os.makedirs(directory, exist_ok=True)
modelName = f"Pre-Trained 1D-Conv - N_{augmentedDataCount} Size_{setSize} Epoch_{epochs} Batch_{batchSize}.keras"
convModel.save(os.path.join(directory, modelName))
print("Saved " + os.path.join(directory, modelName) + "!")

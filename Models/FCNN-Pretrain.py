# Pre-training Fully Connected Neural Network Model using SVG InterExtra augmented data
# Pre-trained on both interpolated and extrapolated data

import numpy as np
from keras.src.layers import BatchNormalization, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.regularizers import l1_l2
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

    # 8 input nodes, into Hidden Layer 1
    model.add(Dense(4, input_dim=8, kernel_initializer=RandomNormal(), kernel_regularizer=l1_l2(l1=1e-3, l2=1e-2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    # Hidden Layer 2
    model.add(Dense(4, kernel_initializer=RandomNormal(), kernel_regularizer=l1_l2(l1=1e-3, l2=1e-2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Output Layer
    model.add(Dense(1, kernel_initializer=RandomNormal(), kernel_regularizer=l1_l2(l1=1e-3, l2=1e-2)))
    model.add(BatchNormalization())
    model.add(Activation('linear'))

    model.compile(optimizer=Adam(learning_rate=learningRate), loss='mean_squared_error')
    return model


datasetModels  = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
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

# Construct FCNN Model
fcnnModel = constructModelLayers()

# Pre-train FCNN
history = fcnnModel.fit(xScaled, y, epochs= epochs, batch_size = batchSize, verbose = 1)

# Print history
# print("Training Loss:", history.history['loss'])

# Save FCNN
directory = os.path.join("Pre-Trained Neural Networks", "FCNN", datasetModels, output)
os.makedirs(directory, exist_ok=True)
modelName = f"Pre-Trained FCNN - N_{augmentedDataCount} Size_{setSize} Epoch_{epochs} Batch_{batchSize}.keras"
fcnnModel.save(os.path.join(directory, modelName))
print("Saved " + os.path.join(directory, modelName) + "!")

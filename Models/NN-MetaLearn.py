# Optimization-based meta-learning for neural network models using SVR, BRR, and GPR augmented data

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import pandas as pd
import config
import os

# Select hyperparameters from config
innerLearningRate = config.innerLearningRate
metaStepSize = config.metaStepSize # Same as outer-loop learning rate, interchangeable terms
metaBatchSize = config.metaBatchSize
metaTasks = config.metaTasks
metaEpochs = config.metaEpochs

nnEpoch = config.nnEpoch
nnBatch = config.nnBatch
nn = config.nn

adSize = config.adSize
adData = config.adData
adYIndex = config.adYIndex

output = "Film Thickness" if adYIndex == -2 else "NTi"
datasetModels  = "Dataset 1 Models" if "Dataset 1" in adData else "Dataset 2 Models"

# Load and normalize augmented data
augmentedData = pd.read_csv(f"Regression Model Data and Metrics/{datasetModels}/{output}/Merged/Merged Size_{adSize} Augmented Data.csv")
x = augmentedData.iloc[:, :-1].values
y = augmentedData.iloc[:, -1].values

# Normalize data
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xLog = np.log1p(x)
xScaled = dataScaler.fit_transform(xLog)

# Load pre-trained neural network, set optimizer
nnModelPath = f"Pretrained Neural Networks/{nn}/{datasetModels}/{output}/Pre-Trained NN - Size_{adSize} Epoch_{nnEpoch} Batch_{nnBatch}.keras"
nnModel = load_model(nnModelPath)
optimizer = Adam(learning_rate=innerLearningRate)

# Meta-Iteration Loop
for metaIter in range(metaTasks):
    # Save temporary weights
    oldWeights = nnModel.get_weights()
    # Get mini batch
    miniBatchIndices = np.random.choice(len(xScaled), metaBatchSize, replace=False)
    xBatchScaled = xScaled[miniBatchIndices]
    yBatch = y[miniBatchIndices]
    # Inner Loop Training (x does not get used)
    for x in range(metaEpochs):
        # Record computations, to later compute gradients/derivatives
        with tf.GradientTape() as tape:
            # Calculate LMSE of predicted and actual output
            predictions = nnModel(xBatchScaled)
            lmseLoss  = tf.keras.losses.mean_squared_error(yBatch, predictions)
        # Compute gradients/derivatives of MSE equation, tells us direction/magnitude to minimize loss. Multiply by loss function, add onto weight
        gradients = tape.gradient(lmseLoss, nnModel.trainable_weights)
        # Pair gradients and trainable weights, updates weights of NN by adding (innerLearningRate)*(gradients) to weights
        optimizer.apply_gradients(zip(gradients, nnModel.trainable_weights))
    # Apply Meta-Update
    newWeights = nnModel.get_weights()
    for var in range(len(newWeights)):
        newWeights[var] = oldWeights[var] + ((newWeights[var] - oldWeights[var]) * metaStepSize)
    nnModel.set_weights(newWeights)
    # Logging loss every 100 iterations
    if metaIter % 100 == 0:
        print(f"Meta-iteration {metaIter}: Loss = {np.mean(lmseLoss.numpy()):.6f}")

# Save trained model
directory = f"Meta-Trained Neural Networks/{nn}/{datasetModels}/{output}/"
os.makedirs(directory, exist_ok=True)
trainedModelName = f"Meta-Trained NN - Size_{adSize} Epoch_{nnEpoch} Batch_{nnBatch}.keras"
nnModel.save(directory + trainedModelName)
print("Saved " + directory + trainedModelName + "!")

# Visualize results - to be added

# ================================================================================
# Extra code to be considered
# ================================================================================

# Auto-adjusting outer learning rate/step size that makes sure updates start big and get smaller over time
# fractionDone = metaIter / metaTasks
#     currentMetaStepSize = (1 - fractionDone) * metaStepSize
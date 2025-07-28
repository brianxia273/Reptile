# Optimization-based first-order meta-learning for neural network models using SVR, BRR, and GPR augmented data

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import pandas as pd
import config
import os
import time

# Select hyperparameters from config
innerStepSize = config.p3InnerStepSize
metaStepSize = config.p3MetaStepSize  # Same as outer-loop learning rate, interchangeable terms
metaBatchSize = config.p3MetaBatchSize
metaTasks = config.p3MetaTasks
metaEpochs = config.p3MetaEpochs

nnSize = config.p3NNSize
nnEpoch = config.p3NNEpoch
nnBatch = config.p3NNBatch
nn = "FCNN"

adSize = config.p3Size
adData = os.path.join("Datasets", config.p3Data)
adYIndex = config.p3YIndex
augmentedDataCount = config.p3N
randomState = config.p3RandomState

seed = config.p3seed
np.random.seed(seed)
tf.random.set_seed(seed)

output = "Film Thickness" if adYIndex == -2 else "NTi"
datasetModels = "Dataset 1 Models" if "Dataset 1" in adData else "Dataset 2 Models"

# Load and normalize 3 augmented datasets

# SVR
svrDataDirectory = os.path.join("Regression Model Data and Metrics", datasetModels, output, "SVR", f"SVR MetaTrain N_{augmentedDataCount} Size_{adSize} Random_{randomState} Augmented Data.csv")
svrAugData = pd.read_csv(svrDataDirectory)
svrX = svrAugData.iloc[:, :-1].values
svrY = svrAugData.iloc[:, -1].values # Always 1 output col in aug data
svrDataScaler = MinMaxScaler(feature_range=(-1, 1))
svrXLog = np.log1p(svrX)
svrXScaled = svrDataScaler.fit_transform(svrXLog)

# BRR
brrDataDirectory = os.path.join("Regression Model Data and Metrics", datasetModels, output, "BRR", f"BRR N_{augmentedDataCount} Size_{adSize} Random_{randomState} Augmented Data.csv")
brrAugData = pd.read_csv(brrDataDirectory)
brrX = brrAugData.iloc[:, :-1].values
brrY = brrAugData.iloc[:, -1].values
brrDataScaler = MinMaxScaler(feature_range=(-1, 1))
brrXLog = np.log1p(brrX)
brrXScaled = brrDataScaler.fit_transform(brrXLog)

# GPR
gprDataDirectory = os.path.join("Regression Model Data and Metrics", datasetModels, output, "GPR", f"GPR N_{augmentedDataCount} Size_{adSize} Random_{randomState} Augmented Data.csv")
gprAugData = pd.read_csv(gprDataDirectory)
gprX = gprAugData.iloc[:, :-1].values
gprY = gprAugData.iloc[:, -1].values
gprDataScaler = MinMaxScaler(feature_range=(-1, 1))
gprXLog = np.log1p(gprX)
gprXScaled = gprDataScaler.fit_transform(gprXLog)

# Load pre-trained neural network, set optimizer
nnModelPath = os.path.join("Pre-Trained Neural Networks", nn, datasetModels, output,
                           f"Pre-Trained {nn} - N_{augmentedDataCount} Size_{adSize} Epoch_{nnEpoch} Batch_{nnBatch}.keras")
nnModel = load_model(nnModelPath)
optimizer = Adam(learning_rate=innerStepSize)

trainedModelName = f"Meta-Trained {nn} - N_{augmentedDataCount} Size_{adSize} Epoch_{nnEpoch} Batch_{nnBatch}.keras"
print("Training " + trainedModelName + f", randomState {randomState}")
mse = MeanSquaredError()
startTime = time.time()

@tf.function
def innerLoop(xTensor, yTensor):
    # Record MSE computations, to later compute gradients/derivatives
    with tf.GradientTape() as tape:
        # Calculate LMSE of predicted and actual output
        predictions = nnModel(xTensor)
        lmseLoss = mse(yTensor, predictions)
    # Compute gradients/derivatives of MSE equation, tells us direction/magnitude to minimize loss. Multiply by loss function, add onto weight
    gradients = tape.gradient(lmseLoss, nnModel.trainable_weights)
    # Pair gradients and trainable weights, updates weights of NN by adding (innerLearningRate)*(gradients) to weights
    optimizer.apply_gradients(zip(gradients, nnModel.trainable_weights))
    return lmseLoss

# Meta-Iteration Loop
for metaIter in range(metaTasks):
    # Save temporary weights
    oldWeights = nnModel.get_weights()
    # Choose SVR, BRR, or GPR
    datasetChoice = np.random.randint(0, 3)
    xScaled = svrXScaled if datasetChoice == 0 else brrXScaled if datasetChoice == 1 else gprXScaled
    y = svrY if datasetChoice == 0 else brrY if datasetChoice == 1 else gprY
    # Get mini batch
    miniBatchIndices = np.random.choice(len(xScaled), metaBatchSize, replace=False)
    xBatchScaled = xScaled[miniBatchIndices]
    yBatch = y[miniBatchIndices]
    # Inner Loop Training
    xTensor, yTensor = tf.convert_to_tensor(xBatchScaled, dtype=tf.float32), tf.convert_to_tensor(yBatch, dtype=tf.float32)
    for _ in range(metaEpochs):
        lmseLoss = innerLoop(xTensor, yTensor)
    # Apply Meta-Update
    newWeights = nnModel.get_weights()
    for var in range(len(newWeights)):
        newWeights[var] = oldWeights[var] + ((newWeights[var] - oldWeights[var]) * metaStepSize)
    nnModel.set_weights(newWeights)
    # Logging loss every 100 iterations
    if metaIter % 100 == 0:
        print(f"Meta-iteration {metaIter}: Loss = {np.mean(lmseLoss.numpy()):.6f}")

# Save trained model
modelDirectory = os.path.join("Meta-Trained Neural Networks", nn, datasetModels, output)
os.makedirs(modelDirectory, exist_ok=True)
nnModel.save(os.path.join(modelDirectory, trainedModelName))
print("Saved " + os.path.join(modelDirectory, trainedModelName) + "!")
endTime = time.time()
print(f"Time Elapsed: {endTime - startTime} seconds")

# Visualize results - to be added
# ================================================================================
# Extra reference code
# ================================================================================

# Auto-adjusting outer learning rate/step size that makes sure updates start big and get smaller over time
# fractionDone = metaIter / metaTasks
#     currentMetaStepSize = (1 - fractionDone) * metaStepSize

# # Saving Data Scaler
# scalerDirectory = os.path.join("Data Scalers", nn, datasetModels, output)
# os.makedirs(scalerDirectory, exist_ok=True)
# scalerName = f"Meta-Trained {nn} - N_{augmentedDataCount} Size_{adSize} Epoch_{nnEpoch} Batch_{nnBatch} DataScaler.pkl"
# joblib.dump(dataScaler, os.path.join(scalerDirectory, scalerName))
# print("Saved " + os.path.join(scalerDirectory, scalerName) + "!")

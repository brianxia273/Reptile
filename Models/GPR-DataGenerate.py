# Generating augmented data with GPR regression model
# NOTE: MUST ADJUST GPR HYPERPARAMETERS FOR BEST PERFORMANCE
# NOTE: INTERPOLATED/EXTRAPOLATED DATA RATIO IS APPROXIMATED, NOT A PRECISE RATIO

import numpy as np
import pandas as pd
import os
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import config
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Select size, dataset, output, and randomState from config
setSize = config.p1Size
data = os.path.join("Datasets", config.p1Data)
yIndex = config.p1YIndex
randomState = config.p1RandomState
model = "GPR"
augmentedDataCount = config.p1N

# Automating file creation
datasetModels = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
output = "Film Thickness" if yIndex == -2 else "NTi"

directory = os.path.join("Regression Model Data and Metrics", datasetModels, output, model)
os.makedirs(directory, exist_ok=True)
df = pd.read_csv(data)
x = df.iloc[:, :-2].values
# Selecting output
y = df.iloc[:, yIndex].values

# 80% data to train, 20% leave for testing. random_state is set in config
trainSize = min(setSize, int(0.8 * len(x)), len(x))
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=randomState)

# Scaling data
xTrainLog = np.log1p(xTrain)
xTestLog = np.log1p(xTest)
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xTrainScaled = dataScaler.fit_transform(xTrainLog)
xTestScaled = dataScaler.transform(xTestLog)

# Init GPR model
gprKernel = ConstantKernel(1.0) * Matern(length_scale=50, nu=0.5)
gpr = GaussianProcessRegressor(alpha=0.01, kernel=gprKernel, n_restarts_optimizer=10, normalize_y=True,
                               optimizer="fmin_l_bfgs_b")
gpr.fit(xTrainScaled, yTrain)

# Interpolation
xMin = x.min(axis=0)
xMax = x.max(axis=0)
totalAugmentedX = augmentedDataCount
# Scaling Data
xAugmented = np.random.uniform(xMin, xMax, size=(totalAugmentedX, x.shape[1]))
xAugmentedLog = np.log1p(xAugmented)
xAugmentedScaled = dataScaler.transform(xAugmentedLog)
yAugmented = gpr.predict(xAugmentedScaled)
xColumns = np.array(xAugmented)
yColumn = np.array(yAugmented)

dfCSV = pd.DataFrame(np.column_stack((xColumns, yColumn)))
saveDirectory = os.path.join(directory,
                             f"{model} N_{augmentedDataCount} Size_{setSize} Random_{randomState} Augmented Data.csv")
dfCSV.to_csv(saveDirectory, index=False, header=False)

print(f"Finished {model} N_{augmentedDataCount} Size_{setSize} Random_{randomState} Augmented Data!")

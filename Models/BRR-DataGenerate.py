# Generating augmented data with BRR regression model

import numpy as np
import pandas as pd
import os
import config
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Select size, dataset, output, and randomState from config
setSize = config.size
data = os.path.join("Datasets" , config.data)
yIndex = config.yIndex
randomState = config.randomState
model = "BRR"
extrapolationRange = config.extrapolationRange

# Automating file creation
datasetModels  = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
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

# Init BRR model
brr = BayesianRidge()
brr.fit(xTrainScaled, yTrain)

# Interpolation
xMin = x.min(axis=0)
xMax = x.max(axis=0)
# BRR will not extrapolate
# xMin = xMin - extrapolationRange * (xMax - xMin)
# xMax = xMax + extrapolationRange * (xMax - xMin)
totalAugmentedX = 1028
# 1028 points total
xAugmented = np.random.uniform(xMin, xMax, size=(totalAugmentedX, x.shape[1]))
xAugmentedLog = np.log1p(xAugmented)
xAugmentedScaled = dataScaler.transform(xAugmentedLog)
yAugmented = brr.predict(xAugmentedScaled)
xColumns = np.array(xAugmented)
yColumn = np.array(yAugmented)

dfCSV = pd.DataFrame(np.column_stack((xColumns, yColumn)))
saveDirectory = os.path.join(directory, f"{model} Size_{setSize} Random_{randomState} Augmented Data.csv")
dfCSV.to_csv(saveDirectory, index= False, header=False)

print(f"Saved {model} Size_{setSize} Random_{randomState} Augmented Data!")




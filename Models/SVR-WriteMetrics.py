# Writing metrics for SVR regression model for sizes 5-40.
# NOTE: MUST ADJUST SVG HYPERPARAMETERS FOR BEST PERFORMANCE

import numpy as np
import pandas as pd
import joblib
import os
import config
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, explained_variance_score

# Select dataset, output, and randomState from config
setSize = 40
data = os.path.join("Datasets", config.p1Data)
yIndex = config.p1YIndex
randomState = config.p1RandomState
model = "SVR"

# Automating file creation
datasetModels = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
output = "Film Thickness" if yIndex == -2 else "NTi"

directory = os.path.join("Regression Model Data and Metrics", datasetModels, output, model)
os.makedirs(directory, exist_ok=True)
with open(os.path.join(directory, f"{model} Random_{randomState} Metric Iteration Evaluation.txt"), "w") as f:
    # Write headers
    f.write("MSE, RMSE, MAPE, EV, and R^2 Metrics\n")
    f.write(f"Current Model Dataset: {data}\n")
    f.write(f"Output Variable: {output}\n")
    f.write(f"Random State: {randomState}\n")
    f.write("=" * 50 + "\n")
    while setSize != 0:
        df = pd.read_csv(data)
        x = df.iloc[:, :-2].values
        y = df.iloc[:, yIndex].values  # Selecting output

        # 80% data to train, 20% leave for testing. random_state is set in config
        trainSize = min(setSize, int(0.8 * len(x)), len(x))
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=randomState)

        # Scaling data
        xTrainLog = np.log1p(xTrain)
        xTestLog = np.log1p(xTest)
        dataScaler = MinMaxScaler(feature_range=(-1, 1))
        xTrainScaled = dataScaler.fit_transform(xTrainLog)
        xTestScaled = dataScaler.transform(xTestLog)

        # Init SVR model
        svr = SVR(kernel='linear', C=.1, epsilon=0.05, gamma='scale')  # ADJUST HYPERPARAMETERS
        svr.fit(xTrainScaled, yTrain)

        # Initial predictions
        yPredict = svr.predict(xTestScaled)
        mseCurrent = mean_squared_error(yTest, yPredict)
        rmseCurrent = np.sqrt(mseCurrent)
        mapeCurrent = np.mean(np.abs((yTest - yPredict) / yTest))
        evCurrent = explained_variance_score(yTest, yPredict)
        currentModelScore = svr.score(xTestScaled, yTest)

        # Write metrics
        f.write(f"Current Model Training Size: {setSize}\n")
        f.write(f"MSE: {mseCurrent}\n")
        f.write(f"RMSE: {rmseCurrent}\n")
        f.write(f"MAPE: {mapeCurrent}\n")
        f.write(f"EV: {evCurrent}\n")
        f.write(f"R^2: {currentModelScore}\n")
        f.write("-" * 50 + "\n")
        print(f"Completed {setSize}!")

        # Saving trained model
        # directory = f"Regression Model Data and Metrics/Starter Models/{datasetModels}/{output}/{model}/"
        # modelName = f"{model.lower()}_model_{setSize}.pkl"
        # os.makedirs(directory, exist_ok=True)
        # joblib.dump(svr, os.path.join(directory, modelName))
        # print("Saved!")

        setSize -= 5

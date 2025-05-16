# GridSearch and graph for best hyperparameters of SVR regression model for different randomStates

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import config
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,  explained_variance_score
import os

# Select size, dataset, output, and randomState from config
setSize = config.p1Size
data = os.path.join("Datasets" , config.p1Data)
yIndex = config.p1YIndex
randomState = config.p1RandomState

# Selecting dataset columns
df = pd.read_csv(data)
x = df.iloc[:, :-2].values
# Selecting output
y = df.iloc[:, yIndex].values

# 80% data to train, 20% leave for testing.
trainSize = min(setSize, int(0.8 * len(x)), len(x))
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=randomState)

# Scaling data
xTrainLog = np.log1p(xTrain)
xTestLog = np.log1p(xTest)
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xTrainScaled = dataScaler.fit_transform(xTrainLog)
xTestScaled = dataScaler.transform(xTestLog)

# GridSearchCV finding optimal hyperparameters, with cross-validation
paramGrid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000, 2000, 5000, 5010],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto',  0.01, 0.1, 1, 10, 100,],
    'epsilon': [0.00009, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
}
gridSearch = GridSearchCV(SVR(), paramGrid, cv=5, scoring='r2', n_jobs=-1)
gridSearch.fit(xTrainScaled, yTrain)
print("Best SVR Parameters:", gridSearch.best_params_)
bestSVR = gridSearch.best_estimator_
trainScore = bestSVR.score(xTrainScaled, yTrain)
print("Train Set Score (R^2):", trainScore)
testScore = bestSVR.score(xTestScaled, yTest)
print("Test Set Score (R^2):", testScore)

# SVR model making predictions
yPredict = bestSVR.predict(xTestScaled)
mseCurrent = mean_squared_error(yTest, yPredict)
rmseCurrent = np.sqrt(mseCurrent)
mapeCurrent = np.mean(np.abs((yTest - yPredict) / yTest))
evCurrent = explained_variance_score(yTest, yPredict)
currentModelScore = bestSVR.score(xTestScaled, yTest)
print("Current Model Dataset:", data)
print("Current Model Training Size:",setSize)
print("Random State:",randomState)
print("Current Model MSE:", mseCurrent)
print("Current Model RMSE:", rmseCurrent)
print("Current Model MAPE:", mapeCurrent)
print("Current Model EV:", evCurrent)
print("Current Model R^2:", currentModelScore)

# Plotting data
plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")
sns.scatterplot(x=yTest, y=yPredict, color="blue", s=50, edgecolor='black', alpha=0.75)
min_val = min(min(yTest), min(yPredict))
max_val = max(max(yTest), max(yPredict))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Fit (y = x)")
plt.title("SVR Model - " + ("Film-Thickness" if yIndex == -2 else "N/Ti Ratio"), fontsize=16)
plt.xlabel("Measurements", fontsize=14)
plt.ylabel("SVR Predictions", fontsize=14)
plt.legend()
plt.show()

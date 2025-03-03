# Replica of SVR model from Reactive Sputtering study

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,  explained_variance_score


# Selecting dataset
data = "Datasets/NitrideMetal (Dataset 2 Models) NTi.csv"
df = pd.read_csv(data)

x = df.iloc[:, :-2].values
# Selecting output
# A y-index of -2 = film-thickness, -1 = N/Ti ratio
yIndex = -2
y = df.iloc[:, yIndex].values

# 80% data to train, 20% leave for testing. random_state is 40 temporarily, will look for the study's random_state later
setSize = 40
trainSize = min(setSize, int(0.8 * len(x)), len(x))
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=40)


# Scaling data
xTrainLog = np.log1p(xTrain)
xTestLog = np.log1p(xTest)
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xTrainScaled = dataScaler.fit_transform(xTrainLog)
xTestScaled = dataScaler.transform(xTestLog)

# Training SVR model, determining epsilon tube,
svr = SVR(kernel='rbf', C=1000.0, epsilon=0.5, gamma='scale')
svr.fit(xTrainScaled, yTrain)

# GridSearchCV finding optimal hyperparameters, with cross-validation
paramGrid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000, 2000, 5000, 5010],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto',  0.01, 0.1, 1, 10, 100,],
    'epsilon': [0.00009, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
}
gridSearch = GridSearchCV(svr, paramGrid, cv=5, scoring='r2', n_jobs=-1)
gridSearch.fit(xTrainScaled, yTrain)
print("Best SVR Parameters:", gridSearch.best_params_)
bestSvr = gridSearch.best_estimator_
testScore = bestSvr.score(xTestScaled, yTest)
print("Test Set Score (R^2):", testScore)

# SVR model making predictions
yPredict = svr.predict(xTestScaled)
mseCurrent = mean_squared_error(yTest, yPredict)
rmseCurrent = np.sqrt(mseCurrent)
mapeCurrent = np.mean(np.abs((yTest - yPredict) / yTest))
evCurrent = explained_variance_score(yTest, yPredict)
currentModelScore = svr.score(xTestScaled, yTest)
print("Current Model Dataset:", data)
print("Current Model Training Size:",setSize)
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

# Saving trained model
# os.makedirs("Saved Models", exist_ok=True)
# joblib.dump(svr, "Saved Models/Starter Models/svr_model.pkl")
# print("Saved!")

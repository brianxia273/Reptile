# Replica of GPR model from Reactive Sputtering study

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
import config
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, ConstantKernel, WhiteKernel, Matern, RationalQuadratic, ExpSineSquared
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
#something ssrnejn
# Select size, dataset, output, and randomState from config
setSize = config.size
data = config.data
yIndex = config.yIndex
randomState = config.randomState
model = config.model

# Automating file creation
datasetModels  = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
output = "Film Thickness" if yIndex == -2 else "NTi"


directory = f"Saved Models/{datasetModels}/{output}/{model}"
os.makedirs(directory, exist_ok=True)
with open(f"Saved Models/{datasetModels}/{output}/{model}/{model} Metric Iteration Evaluation.txt", "w") as f:
    # Write headers
    f.write("MSE, RMSE, MAPE, EV, and R^2 Metrics\n")
    f.write(f"Current Model Dataset: {data}\n")
    f.write(f"Output Variable: {output}\n")
    f.write("=" * 50 + "\n")
    while setSize != 0:
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
        #dataScaler = StandardScaler()
        xTrainScaled = dataScaler.fit_transform(xTrainLog)
        xTestScaled = dataScaler.transform(xTestLog)

        # Init GPR model

        #kernel = RBF(length_scale=3.75)

        #kernel = ConstantKernel(1.0, (1e-4, 1e1)) * RBF(length_scale=1.0, length_scale_bounds=(1e-4, 1e1))
        kernel = ConstantKernel(1.0) * (RationalQuadratic() + RBF() + ExpSineSquared())

        # Define the parameter grid without kernel combinations
        param_grid = {
            "kernel": [
                ConstantKernel(1.0) * Matern(length_scale=50, nu=1.5),
                # Matern kernel with length_scale=50, nu=1.5 (previously successful)
                ConstantKernel(1.0) * Matern(length_scale=75, nu=1.5),
                ConstantKernel(1.0) * Matern(length_scale=60, nu=1.5),
                ConstantKernel(1.0) * Matern(length_scale=40, nu=1.5),
                ConstantKernel(1.0) * Matern(length_scale=55, nu=3.0),
                ConstantKernel(1.0) * Matern(length_scale=50, nu=0.5),
                ConstantKernel(1.0) * Matern(length_scale=50, nu=2.5),
            ],
            "alpha": [1e-4, 1e-2, 1e-3],
            "n_restarts_optimizer": [100, 10, 500, 550],
            "normalize_y": [True],
            "optimizer": ["fmin_l_bfgs_b"],
        }

        # Define the GaussianProcessRegressor
        gpr = GaussianProcessRegressor(kernel = kernel)

        grid_search = GridSearchCV(gpr, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
        grid_search.fit(xTrainScaled, yTrain)
        print("Best parameters:", grid_search.best_params_)
        gpr_best = grid_search.best_estimator_
        gpr_best.fit(xTrainScaled, yTrain)

        # Initial predictions
        yPredict = gpr_best.predict(xTestScaled)
        mseCurrent = mean_squared_error(yTest, yPredict)
        rmseCurrent = np.sqrt(mseCurrent)
        mapeCurrent = np.mean(np.abs((yTest - yPredict) / yTest))
        evCurrent = explained_variance_score(yTest, yPredict)
        currentModelScore = gpr_best.score(xTestScaled, yTest)

        # Writing metrics
        f.write(f"Current Model Training Size: {setSize}\n")
        f.write(f"MSE: {mseCurrent}\n")
        f.write(f"RMSE: {rmseCurrent}\n")
        f.write(f"MAPE: {mapeCurrent}\n")
        f.write(f"EV: {evCurrent}\n")
        f.write(f"R^2: {currentModelScore}\n")
        f.write("-" * 50 + "\n")
        print(f"Completed {setSize}!")

        #Saving trained model
        directory = f"Saved Models/Starter Models/{datasetModels}/{output}/{model}/"
        modelName = f"{model.lower()}_model_{setSize}.pkl"
        os.makedirs(directory, exist_ok=True)
        joblib.dump(gpr_best, os.path.join(directory, modelName))
        print("Saved!")
        setSize -= 5



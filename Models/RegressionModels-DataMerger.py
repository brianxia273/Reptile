# Merging SVR, BRR, and GPR augmented datasets into a mixed dataset
# NOTE: MUST MANUALLY SELECT PARAMETERS, DOES NOT USE CONFIG
# NOTE: ONLY USES INTERPOLATED DATA

import pandas as pd
import os

# ================================================================================
# Choose randomStates of datasets to merge. Choose size, dataset, and output from config
svrDataRS: int = 47
brrDataRS: int = 47
gprDataRS: int = 44
# ================================================================================
# Select size, dataset, and output
setSize: int = 40
data: str = "Nitride (Dataset 1) NTi.csv"
yIndex: int = -2               # (-2) = film-thickness, (-1) = N/Ti ratio

# AVAILABLE DATASETS:
    # FullData.csv
    # Metal (Alone).csv
    # Metal (Alone) NTi.csv
    # Nitride (Dataset 1).csv
        # Nitride (Dataset 1) NTi.csv
    # NitrideMetal (Dataset 2).csv
        # NitrideMetal (Dataset 2) NTi.csv
# ================================================================================

data = os.path.join("Datasets", data)
datasetModels  = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
output = "Film Thickness" if yIndex == -2 else "NTi"
svrDirectory = os.path.join("Regression Model Data and Metrics", datasetModels, output, "SVR", f"SVR Size_{setSize} Random_{svrDataRS} Augmented Data.csv")
brrDirectory = os.path.join("Regression Model Data and Metrics", datasetModels, output, "BRR", f"BRR Size_{setSize} Random_{brrDataRS} Augmented Data.csv")
gprDirectory = os.path.join("Regression Model Data and Metrics", datasetModels, output, "GPR", f"GPR Size_{setSize} Random_{gprDataRS} Augmented Data.csv")

dataframes = [pd.read_csv(svrDirectory, header=None), pd.read_csv(brrDirectory, header=None), pd.read_csv(gprDirectory, header=None)]
mergedDF = pd.concat(dataframes, axis=0, ignore_index=True)
mixedMergedDF = mergedDF.sample(frac=1, random_state=45).reset_index(drop=True)

mergedDirectory = os.path.join("Regression Model Data and Metrics", datasetModels, output, "Merged")
os.makedirs(mergedDirectory, exist_ok=True)
csvCounter = 1
outputCSV = f"Merged #{csvCounter} Size_{setSize} Augmented Data.csv"
while os.path.exists(os.path.join(mergedDirectory, outputCSV)):
    csvCounter += 1
    outputCSV = f"Merged #{csvCounter} Size_{setSize} Augmented Data.csv"
mixedMergedDF.to_csv(os.path.join(mergedDirectory, outputCSV), index=False, header=False)
print(f'Merged data saved to {os.path.join(mergedDirectory, outputCSV)}!')
with open(os.path.join(mergedDirectory, "Merged RandomState Log"), "a") as f:
    f.write(f"Merged Data Size_{setSize} #{csvCounter} = SVR({svrDataRS}), BRR({brrDataRS}), GPR({gprDataRS})\n")
    f.write("-" * 50 + "\n")
print(f'RandomStates logged in Merged RandomState Log!')

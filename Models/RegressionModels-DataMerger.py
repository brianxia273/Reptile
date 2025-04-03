# Merging SVR, BRR, and GPR augmented datasets into a mixed dataset
# NOTE: MUST MANUALLY SELECT PARAMETERS, DOES NOT USE CONFIG
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
data: str = "Datasets/Nitride (Dataset 1) NTi.csv"
yIndex: int = -2               # (-2) = film-thickness, (-1) = N/Ti ratio

# AVAILABLE DATASETS:
    # Datasets/FullData.csv
    # Datasets/Metal (Alone).csv
    # Datasets/Metal (Alone) NTi.csv
    # Datasets/Nitride (Dataset 1).csv
        # Datasets/Nitride (Dataset 1) NTi.csv
    # Datasets/NitrideMetal (Dataset 2).csv
        # Datasets/NitrideMetal (Dataset 2) NTi.csv
# ================================================================================

datasetModels  = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
output = "Film Thickness" if yIndex == -2 else "NTi"
svrDirectory = f"Regression Model Data and Metrics/{datasetModels}/{output}/SVR/SVR Size_{setSize} Random_{svrDataRS} Augmented Data.csv"
brrDirectory = f"Regression Model Data and Metrics/{datasetModels}/{output}/BRR/BRR Size_{setSize} Random_{brrDataRS} Augmented Data.csv"
gprDirectory = f"Regression Model Data and Metrics/{datasetModels}/{output}/GPR/GPR Size_{setSize} Random_{gprDataRS} Augmented Data.csv"


dataframes = [pd.read_csv(svrDirectory, header=None), pd.read_csv(brrDirectory, header=None), pd.read_csv(gprDirectory, header=None)]
mergedDF = pd.concat(dataframes, axis=0, ignore_index=True)
mixedMergedDF = mergedDF.sample(frac=1, random_state=45).reset_index(drop=True)

mergedDirectory = f"Regression Model Data and Metrics/{datasetModels}/{output}/Merged/"
os.makedirs(mergedDirectory, exist_ok=True)
csvCounter = 1
outputCSV = f"Merged #{csvCounter} Size_{setSize} Augmented Data.csv"
while os.path.exists(mergedDirectory + outputCSV):
    csvCounter += 1
    outputCSV = f"Merged #{csvCounter} Size_{setSize} Augmented Data.csv"
mixedMergedDF.to_csv(mergedDirectory + outputCSV, index=False, header=False)
print(f'Merged data saved to {mergedDirectory + outputCSV}!')
with open(mergedDirectory + "Merged RandomState Log", "a") as f:
    f.write(f"Merged Data Size_{setSize} #{csvCounter} = SVR({svrDataRS}), BRR({brrDataRS}), GPR({gprDataRS})\n")
    f.write("-" * 50 + "\n")
print(f'RandomStates logged in Merged RandomState Log!')

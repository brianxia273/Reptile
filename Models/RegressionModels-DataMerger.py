import pandas as pd
import os
import config
# ================================================================================
# ================================================================================
# Choose randomStates of datasets to merge
svrDataRS = 1
brrDataRS = 1
gprDataRS = 1
# ================================================================================
# ================================================================================

# Select size, dataset, output, and randomState from config
setSize = config.size
data = config.data
yIndex = config.yIndex

datasetModels  = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
output = "Film Thickness" if yIndex == -2 else "NTi"
svrDirectory = f"Regression Model Data and Metrics/{datasetModels}/{output}/SVR/SVR Size_{setSize} Random_{svrDataRS} Augmented Data.csv"
brrDirectory = f"Regression Model Data and Metrics/{datasetModels}/{output}/BRR/BRR Size_{setSize} Random_{brrDataRS} Augmented Data.csv"
gprDirectory = f"Regression Model Data and Metrics/{datasetModels}/{output}/GPR/GPR Size_{setSize} Random_{gprDataRS} Augmented Data.csv"


dataframes = [pd.read_csv(svrDirectory),pd.read_csv(brrDirectory), pd.read_csv(gprDirectory)]
mergedDF = pd.concat(dataframes, ignore_index=True)
mixedMergedDF = mergedDF.sample(frac=1, random_state=45).reset_index(drop=True)

mergedDirectory = f"Regression Model Data and Metrics/{datasetModels}/{output}/Merged/"
os.makedirs(mergedDirectory, exist_ok=True)
outputCSV = f"Merged Size_{setSize} Augmented Data.csv"
mixedMergedDF.to_csv(mergedDirectory + outputCSV, index=False)
print(f'Merged data saved to {mergedDirectory + outputCSV}!')
with open(mergedDirectory + "Merged RandomState Log", "w") as f:
    f.write(f"Merged Data (Size:{setSize}) = SVR({svrDataRS}), BRR({brrDataRS}), GPR({gprDataRS})\n")
    f.write("-" * 50 + "\n")
print(f'RandomStates logged in Merged RandomState Log!')

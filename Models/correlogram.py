# Making correlogram of input and output variables

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# Selecting dataset
df = pd.read_csv(os.path.join("Datasets", "NitrideMetal (Dataset 2) NTi.csv"))

# Creating correlogram
correlationMatrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlationMatrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlogram of Variables')
plt.show()

# AVAILABLE DATASETS:
    # FullData.csv
    # Metal (Alone).csv
    # Metal (Alone) NTi.csv
    # Nitride (Dataset 1).csv
        # Nitride (Dataset 1) NTi.csv
    # NitrideMetal (Dataset 2).csv
        # NitrideMetal (Dataset 2) NTi.csv
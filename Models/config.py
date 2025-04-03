# ================================================================================
# ================================================================================

# WriteMetrics, DataGenerate, and GridSearch Configuration
size: int = 40
data: str = "Datasets/Nitride (Dataset 1) NTi.csv"
randomState: int = 43
yIndex: int = -2  # (-2) = film-thickness, (-1) = N/Ti ratio
extrapolationRange: float = 0.03

# ================================================================================
# ================================================================================

# Neural Network Pretrain Configuration
nSize: int = 40
nData: str = "Datasets/Nitride (Dataset 1) NTi.csv"
nRandomState: int = 47
nYIndex: int = -2  # (-2) = film-thickness, (-1) = N/Ti ratio
learningRate: float = 0.01
batchSize: int = 1000 # {16, 512, 1028}
epochs: int = 1028 # {20, 200, 1000}

# ================================================================================
# ================================================================================

# Neural Network MetaLearn Configuration
metaTasks: int = 50 # {50, 100, 1000, 2000}
metaStepSize: float = 0.05 # {0.05, 0.2}
metaBatchSize: int = 5 # {5, 20}
metaEpochs: int = 5 # {5, 10, 100, 200}
innerLearningRate: float = 0.01

nnSize: int = 40
nnEpoch: int = 16
nnBatch: int = 20
nn: str = "FCNN"

adSize: int = 40
adData: str = "Datasets/Nitride (Dataset 1) NTi.csv"
adYIndex: int = -2 # (-2) = film-thickness, (-1) = N/Ti ratio

# ================================================================================
# ================================================================================


# AVAILABLE DATASETS:
    # Datasets/FullData.csv
    # Datasets/Metal (Alone).csv
    # Datasets/Metal (Alone) NTi.csv
    # Datasets/Nitride (Dataset 1).csv
        # Datasets/Nitride (Dataset 1) NTi.csv
    # Datasets/NitrideMetal (Dataset 2).csv
        # Datasets/NitrideMetal (Dataset 2) NTi.csv
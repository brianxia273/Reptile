# Configuration file to define parameters for ML pipeline

# ================================================================================
# ================================================================================

# WriteMetrics, DataGenerate, and GridSearch Configuration
size: int = 40
data: str = "Datasets/Nitride (Dataset 1) NTi.csv"
randomState: int = 47
yIndex: int = -2  # (-2) = film-thickness, (-1) = N/Ti ratio

extrapolationRange: float = 0.03

# ================================================================================
# ================================================================================

# Neural Network Pretrain Configuration
nSize: int = 40
nData: str = "Datasets/Nitride (Dataset 1) NTi.csv"
nRandomState: int = 47 # Selecting randomState of SVG augmented data
nYIndex: int = -2  # (-2) = film-thickness, (-1) = N/Ti ratio
learningRate: float = 0.01
epochs: int = 1000 # {20, 200, 1000}
batchSize: int = 1028 # {16, 512, 1028}

# ================================================================================
# ================================================================================

# Neural Network MetaLearn Configuration
metaTasks: int = 2000 # {50, 100, 1000, 2000}
metaStepSize: float = 0.2 # {0.05, 0.2}
metaBatchSize: int = 20 # {5, 20}
metaEpochs: int = 200 # {5, 10, 100, 200}
innerLearningRate: float = 0.01

nnSize = adSize = 40
nnEpoch: int = 1000 # {20, 200, 1000}
nnBatch: int = 1028 # {16, 512, 1028}
nn: str = "FCNN"

adData: str = "Datasets/Nitride (Dataset 1) NTi.csv"
adYIndex: int = -2 # (-2) = film-thickness, (-1) = N/Ti ratio
adNumber: int = 1 # Merged Dataset number

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
# Configuration file to define parameters for ML pipeline

# ================================================================================
# ================================================================================

# WriteMetrics, DataGenerate, and GridSearch Configuration
size: int = 40
data: str = "Nitride (Dataset 1) NTi.csv"
randomState: int = 47
yIndex: int = -2  # (-2) = film-thickness, (-1) = N/Ti ratio

extrapolationRange: float = 0.03

# ================================================================================
# ================================================================================

# Neural Network Pretrain Configuration
nSize: int = 40
nData: str = "Nitride (Dataset 1) NTi.csv"
nRandomState: int = 47 # Selecting randomState of SVG augmented data
nYIndex: int = -2  # (-2) = film-thickness, (-1) = N/Ti ratio
learningRate: float = 0.05 # NEED TO DOUBLE-CHECK
epochs: int = 1000 # {20, 200, 1000}
batchSize: int = 1028 # {16, 512, 1028}

# ================================================================================
# ================================================================================

# Neural Network MetaLearn Configuration
metaTasks: int = 2000 # {50, 100, 1000, 2000}
metaStepSize: float = 0.2 # {0.05, 0.2}
metaBatchSize: int = 20 # {5, 20}
metaEpochs: int = 200 # {5, 10, 100, 200}
innerLearningRate: float = 0.01 # NEED TO DOUBLE-CHECK

nnSize = adSize = 40
nnEpoch: int = 1000 # {20, 200, 1000}
nnBatch: int = 1028 # {16, 512, 1028}
nn: str = "FCNN"

adData: str = "Nitride (Dataset 1) NTi.csv"
adYIndex: int = -2 # (-2) = film-thickness, (-1) = N/Ti ratio
adNumber: int = 1 # Merged Dataset number (1, 2, 3, etc.)

# ================================================================================
# ================================================================================

# Neural Network FineTune Configuration

ftLearningRate: float = 0.2 # {0.05, 0.2}
ftEpochs: int = 200 # {5, 10, 100, 200}
ftYIndex: int = -2 # (-2) = film-thickness, (-1) = N/Ti ratio
ftData: str = "Nitride (Dataset 1) NTi.csv"
ftBatchSize: int = 20 # {5, 20}

ftNN: str = "FCNN"
ftNNSize: int = 40
ftNNEpochs: int = 1000 # {20, 200, 1000}
ftNNBatch: int = 1028 # {16, 512, 1028}

# ================================================================================
# ================================================================================

# Neural Network TestAccuracy Configuration

taData: str = "Nitride (Dataset 1) NTi.csv"
taYIndex: int = -2 # (-2) = film-thickness, (-1) = N/Ti ratio

# Need to adjust epoch and batch
taNN: str = "FCNN"
taNNSize: int = 40
taNNEpochs: int = 1000 # {20, 200, 1000}
taNNBatch: int = 1028 # {16, 512, 1028}


# ================================================================================
# ================================================================================

# AVAILABLE DATASETS:
    # FullData.csv
    # Metal (Alone).csv
    # Metal (Alone) NTi.csv
    # Nitride (Dataset 1).csv
        # Nitride (Dataset 1) NTi.csv
    # NitrideMetal (Dataset 2).csv
        # NitrideMetal (Dataset 2) NTi.csv
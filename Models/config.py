# Configuration file to define parameters for NN and regression model development

# ================================================================================
# ================================================================================

# Phase 1: WriteMetrics, DataGenerate, and GridSearch Configuration
p1Size: int = 40
p1Data: str = "Nitride (Dataset 1) NTi.csv"
p1RandomState: int = 47
p1YIndex: int = -2  # (-2) = film-thickness, (-1) = N/Ti ratio

p1SvrExtrapolationRange: float = 0.03
p1N: int = 6400 # N = Augmented Data Count, {6400, 12800, 25600}

# ================================================================================
# ================================================================================

# Phase 2: FCNN/1D-CONV Pretrain Configuration
p2Size: int = 40
p2Data: str = "Nitride (Dataset 1) NTi.csv"
p2RandomState: int = 47 # Selecting randomState of SVG augmented data
p2YIndex: int = -2  # (-2) = film-thickness, (-1) = N/Ti ratio
p2LearningRate: float = 0.05 # NEED TO DOUBLE-CHECK
p2BatchSize: int = 1028 # {16, 512, 1028}
p2Epochs: int = 1000 # {20, 200, 1000}
p2N: int = 6400 # N = Augmented Data Count, {6400, 12800, 25600}

# ================================================================================
# ================================================================================

# Phase 3: NN MetaLearn Configuration
p3MetaTasks: int = 2000 # {50, 100, 1000, 2000}
p3MetaStepSize: float = 0.2 # {0.05, 0.2}
p3MetaBatchSize: int = 20 # {5, 20}
p3MetaEpochs: int = 200 # {5, 10, 100, 200}
p3InnerStepSize: float = 0.01 # NEED TO DOUBLE-CHECK

# Choosing Pre-Trained NN using its parameters
p3NN: str = "FCNN"
p3NNSize = p3Size = 40
p3NNEpoch: int = 1000 # {20, 200, 1000}
p3NNBatch: int = 1028 # {16, 512, 1028}

# Selecting Augmented Data parameters
p3Data: str = "Nitride (Dataset 1) NTi.csv"
p3YIndex: int = -2 # (-2) = film-thickness, (-1) = N/Ti ratio
p3Number: int = 1 # Merged Dataset number (1, 2, 3, etc.)
p3N: int = 6400 # N {6400, 12800, 25600}

# ================================================================================
# ================================================================================

# Phase 4: Neural Network FineTune Configuration

p4LearningRate: float = 0.2 # {0.05, 0.2} - Is same as MetaLearn
p4Epochs: int = 200 # {5, 10, 100, 200} - Is same as MetaLearn
p4YIndex: int = -2 # (-2) = film-thickness, (-1) = N/Ti ratio
p4Data: str = "Nitride (Dataset 1) NTi.csv"
p4BatchSize: int = 20 # {5, 20}

# Choosing Meta-Trained NN using its parameters
p4NN: str = "FCNN"
p4NNSize: int = 40
p4NNEpochs: int = 1000 # {20, 200, 1000}
p4NNBatch: int = 1028 # {16, 512, 1028}
p4N: int = 6400 # N {6400, 12800, 25600}

# ================================================================================
# ================================================================================

# Neural Network TestAccuracy Configuration
#
# taData: str = "Nitride (Dataset 1) NTi.csv"
# taYIndex: int = -2 # (-2) = film-thickness, (-1) = N/Ti ratio
#
# # Need to adjust epoch and batch
# taNN: str = "FCNN"
# taNNSize: int = 40
# taNNEpochs: int = 1000 # {20, 200, 1000}
# taNNBatch: int = 1028 # {16, 512, 1028}


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
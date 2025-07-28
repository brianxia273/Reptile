# Configuration file to define parameters for NN and regression model development

# ================================================================================
# ================================================================================

# Phase 1: WriteMetrics, DataGenerate, and GridSearch Configuration
# NOTE: NEED TO SELF-MODIFY HYPERPARAMETERS FOR GPR AND SVR IN WriteMetrics AND DataGenerate
p1Size: int = 40       # Always set to 40 for WriteMetrics
p1Data: str = "NitrideMetal (Dataset 2) NTi.csv"
p1YIndex: int = -1  # (-2) = film-thickness, (-1) = N/Ti ratio
p1RandomState: int = 68

p1SvrExtrapolationRange: float = 0.03
p1N: int = 25600  # N = Augmented Data Count, {6400, 12800, 25600}

# ================================================================================
# ================================================================================

# Phase 2: FCNN/1D-Conv PreTrain Configuration
p2Size: int = 40
p2Data: str = "NitrideMetal (Dataset 2) NTi.csv"
p2YIndex: int = -1  # (-2) = film-thickness, (-1) = N/Ti ratio
p2LearningRate: float = 0.05  # NEED TO DOUBLE-CHECK
p2BatchSize: int = 1028  # {16, 512, 1028}
p2Epochs: int = 1000  # {20, 200, 1000}
p2N: int = 25600  # N = Augmented Data Count, {6400, 12800, 25600}
p2RandomState: int = 68  # Selecting randomState of SVG augmented data

# ================================================================================
# ================================================================================

# Phase 3: NN MetaTrain Configuration
p3MetaStepSize: float = 0.05  # {0.05, 0.2}
p3MetaEpochs: int = 200  # {5, 10, 100, 200}
p3MetaTasks: int = 2000  # {50, 100, 1000, 2000}
p3MetaBatchSize: int = 20  # {5, 20}
p3InnerStepSize: float = 0.05  # NEED TO DOUBLE-CHECK

# Choosing Pre-Trained NN using its parameters
p3NNSize = p3Size = 40
p3NNEpoch: int = 1000  # {20, 200, 1000}
p3NNBatch: int = 1028  # {16, 512, 1028}

# Selecting Augmented Data parameters
p3Data: str = "NitrideMetal (Dataset 2) NTi.csv"
p3YIndex: int = -1  # (-2) = film-thickness, (-1) = N/Ti ratio
p3N: int = 25600  # N {6400, 12800, 25600}

# Choose randomStates of datasets to select from. SVR, BRR, and GPR randomState must be the same
p3RandomState: int = 68

# Seed to control np and tf RNG; change as needed, but currently is not kept track of
p3seed: int = 44

# ================================================================================
# ================================================================================

# Phase 4: NN FineTune Configuration

p4LearningRate: float = 0.05  # {0.05, 0.2} - Is same as MetaLearn (?)
p4Epochs: int = 1000  # {5, 10, 100, 200} - Is same as MetaLearn (?)
p4YIndex: int = -1  # (-2) = film-thickness, (-1) = N/Ti ratio
p4Data: str = "NitrideMetal (Dataset 2) NTi.csv"
p4BatchSize: int = 1028 # {16, 512, 1028} (?)

# Choosing Meta-Trained NN using its parameters
p4NNSize: int = 40
p4NNEpoch: int = 1000  # {20, 200, 1000}
p4NNBatch: int = 1028  # {16, 512, 1028}
p4N: int = 25600  # N {6400, 12800, 25600}

# Choosing randomState for train/test, must be same as previous phases
p4RandomState: int = 68

# ================================================================================
# ================================================================================

# Phase 5: Neural Network TestAccuracy Configuration

p5Data: str = "NitrideMetal (Dataset 2) NTi.csv"
p5YIndex: int = -1 # (-2) = film-thickness, (-1) = N/Ti ratio

# Choosing Fine-Tuned NN using its parameters
p5NN: str = "1D-Conv" # 1D-Conv or FCNN
p5NNSize: int = 40
p5N: int = 25600  # N {6400, 12800, 25600}
p5NNEpoch: int = 1000  # {5, 10, 100, 200}
p5NNBatch: int = 1028  # {16, 512, 1028}

# Choosing randomState for same train/test, must be same as previous phases
p5RandomState: int = 68

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

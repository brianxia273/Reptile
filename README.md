# **Neural Network Generator for Reactive Sputtering Experiments**
_ReadMe Last Updated: 4/8/25_

## **Overview**
This repository contains tools and models for generating and training neural networks designed to assist with reactive sputtering experiments. The primary goal of this project is to harness machine learning techniques, specifically first-order meta-learning, to predict key outcomes in reactive sputtering processes and to overcome the obstacle of limited data. For meta-learning, we leveraged Reptile, an algorithm developed by OpenAI, to train our neural networks to quickly adapt to new tasks by learning from simulated data from three linear regression models: Support Vector Regression (SVR), Bayesian Ridge Regression (BRR), and Gaussian Process Regression (GPR).

The repository includes tools for:

1. Linear Regression Model Evaluation: Records performance of SVR, BRR, and GPR on given datasets

2. Augmented Data Generation: Creates interpolated and extrapolated data using linear regression models

3. Defining and Pre-Training Neural Network Archetypes: Defines FCNN and 1D CONV neural network models and pre-trains them on SVG augmented data.

4. Meta-Training Neural Networks using Reptile Algorithm: Utilizes the Reptile algorithm to adapt model parameters through few-shot learning 

5. Fine-Tuning for Real-World Usage: Adjusts model parameters on real data, completing the model for practical usage


## Acknowledgments

This project is based on the study **[Deep neural network and meta-learning-based reactive sputtering with small data sample counts]** by **[Jeongsu Lee and Chanwoo Yang]**, published in **[Journal of Manufacturing Systems]**. The work provides insights into reactive sputtering processes and has been foundational in the development of machine learning models for predicting deposition outcomes. We extend this work to further optimize and predict sputtering parameters.

- [Lee, Jeongsu, and Chanwoo Yang]. *“Deep Neural Network and meta-learning-based reactive sputtering with small data sample counts.”*. *Journal of Manufacturing Systems*, vol. 62, Jan. 2022, pp. 703–717, [https://doi.org/10.1016/j.jmsy.2022.02.004. ]

---
## **Project Structure**

Reptile/
ª   .gitattributes
ª   directory_structure.txt
ª   README.md
ª   
+---.idea
ª   ª   .gitignore
ª   ª   misc.xml
ª   ª   modules.xml
ª   ª   REPTILE!.iml
ª   ª   workspace.xml
ª   ª   
ª   +---inspectionProfiles
ª           profiles_settings.xml
ª           
+---Models
    ª   BRR-DataGenerate.py
    ª   BRR-WriteMetrics.py
    ª   config.py
    ª   correlogram.py
    ª   FCNN-Pretrain.py
    ª   GPR-DataGenerate.py
    ª   GPR-GridSearch.py
    ª   GPR-WriteMetrics.py
    ª   NN-FineTune.py
    ª   NN-MetaLearn.py
    ª   RegressionModels-DataMerger.py
    ª   SVR-DataGenerate.py
    ª   SVR-GridSearch.py
    ª   SVR-InterExtraDataGenerate.py
    ª   SVR-WriteMetrics.py
    ª   
    +---Datasets
    ª       FullData.csv
    ª       Metal (Alone) NTi.csv
    ª       Metal (Alone).csv
    ª       Nitride (Dataset 1) NTi.csv
    ª       Nitride (Dataset 1).csv
    ª       NitrideMetal (Dataset 2) NTi.csv
    ª       NitrideMetal (Dataset 2).csv
    ª       
    +---Old Reference Code
    ª       BRR-Display.py
    ª       SVR-OldInit.py
    ª       
    +---Pre-Trained Neural Networks
    ª   +---FCNN
    ª       +---Dataset 1 Models
    ª           +---Film Thickness
    ª                   Pre-Trained NN - Size_40 Epoch_1000 Batch_1028.keras
    ª                   
    +---Regression Model Data and Metrics
    ª   +---Dataset 1 Models
    ª   ª   +---Film Thickness
    ª   ª   ª   +---BRR
    ª   ª   ª   ª       BRR Random_47 Metric Iteration Evaluation.txt
    ª   ª   ª   ª       BRR Size_40 Random_47 Augmented Data.csv
    ª   ª   ª   ª       
    ª   ª   ª   +---GPR
    ª   ª   ª   ª       GPR Random_47 Metric Iteration Evaluation.txt
    ª   ª   ª   ª       GPR Size_40 Random_44 Augmented Data.csv
    ª   ª   ª   ª       
    ª   ª   ª   +---Merged
    ª   ª   ª   ª       Merged #1 Size_40 Augmented Data.csv
    ª   ª   ª   ª       Merged RandomState Log
    ª   ª   ª   ª       
    ª   ª   ª   +---SVR
    ª   ª   ª           SVR InterExtra Size_40 Random_47 Augmented Data.csv
    ª   ª   ª           SVR Random_42 Metric Iteration Evaluation.txt
    ª   ª   ª           SVR Size_40 Random_47 Augmented Data.csv
    ª   ª   ª           
    ª   ª   +---NTi
    ª   +---Dataset 2 Models
    ª   ª   +---Film Thickness
    ª   ª   +---NTi
    ª   +---Starter Models
    ª       +---Dataset 1 Models
    ª       ª   +---Film Thickness
    ª       ª   ª   +---BRR
    ª       ª   ª           brr_model_10.pkl
    ª       ª   ª           
    ª       ª   +---NTi
    ª       ª               
    ª       +---Dataset 2 Models
    ª           +---Film Thickness
    ª           +---NTi                 

Regression Model Data and Metrics, Starter Models, Pre-Trained Neural Networks, Meta-Trained Neural Networks, and Fine-Tuned Neural Network directories and sub-directories are all generated by the Python scripts. Example files and sub-directories are shown above.

## **Organization**

*Adjust parameters in *config.py*, and replace datasets as necessary

## 1. Linear Regression Model Evaluation
- SVR-GridSearch.py
  SVR-WriteMetrics.py
  
- BRR-WriteMetrics.py
  
- GPR-GridSearch.py
  GPR-WriteMetrics.py


## 2. Augmented Data Generation
- SVR-DataGenerate.py
  SVR-InterExtraDataGenerate.py
  
- BRR-DataGenerate.py
  
- GPR-DataGenerate.py
  
- RegressionModels-DataMerger.py


## 3. Defining and Pre-Training Neural Network Archetypes
- FCNN-Pretrain.py


## 4. Meta-Training Neural Networks using Reptile Algorithm
- NN-MetaLearn.py


## 5. Fine-Tuning for Real-World Usage
- NN-FineTune.py

---


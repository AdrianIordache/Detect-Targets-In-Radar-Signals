# Detect Targets In Radar Signals
**Comparing SOTA Convolutional Neural Networks to SOTA Vision Transformers for counting objects in radar signals.**

Kaggle competition organized by the Deep Learning teachers during the Artificial Intelligence master's program, 2nd Year, 1st Semester, 2022.

[**1st Place Solution [Public Leaderboard] - 3rd Place Solution [Private Leaderboard]**](https://www.kaggle.com/competitions/detect-targets-in-radar-signals/leaderboard?tab=public)

## Task & Dataset Description
For this task, we have available a trainset with 15500 samples of labeled with one of five classes from 1 to 5, uniformly distributed, representing the number of objects in that images. 

Each class from 2 to 5 will have 3000 examples and only the first class will contain 500 samples more than the others.

The evaluation phase of this project will be done based on the inference of to 5500 samples using as test metric the accuracy score.

## Benchmark Models
  - [x] EfficientNet B5 (pretrained on Noisy Students)
  - [x] Swin Large Transformer (Image Size: 384, Window Size: 12, Pre-trained on ImageNet)
  - [x] BEiT Large Transformer (Image Size: 224, Pre-trained on ImageNet)
  
## Project Structure
  - Stage-0: Baseline Models (various architectures, augmentations, ideas)
  - Stage-1: Grid Search over large hyper-parameter space for a certain architecture
  - Stage-2: Adding Stochastic Weight Averaging (SWA), Sharpness Aware Minimization (SAM) and reducing the hyper-parameter space
  - Stage-3: Switching to 10 Folds Cross-Validation, Pseudo-Labeling, Combined Embeddings and Final Models Ensembles
  
## Final Solutions
### Multiple Concatenated BackBones with SVM Head
<p align="center" width="100%">
    <img src="https://github.com/AdrianIordache/Detect-Targets-In-Radar-Signals/blob/master/images/multiple_concatenated_backbones.png">
</p>

### Voting Ensamble between NN and SVM heads
<p align="center" width="100%">
    <img src="https://github.com/AdrianIordache/Detect-Targets-In-Radar-Signals/blob/master/images/voting_system.png">
</p>

For a more in-depth description of the solution, feel free to check out the documentation paper. :)

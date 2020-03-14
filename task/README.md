# HEPAutoencoders
Autoencoder compression of ATLAS jet events using PyTorch and fastai.
It strives to be easy to experiment with, but also parallelizable and GPU-friendly in order to aid hyperparameters scans on clusters.
Builds directly on the [code](https://github.com/erwulff/lth_thesis_project) of Eric Wulff. Technical explanations can be found in his [thesis](https://lup.lub.lu.se/student-papers/search/publication/9004751). 

## Associated Files
Can be found at Files and Instructions.txt

## Quick guide
Pre-processing:
The data is 4-dimensional.
All processed data will be placed in processed_data/ after extraction (by default). No normalization or other ML-related pre-processing is done in this step. 

Training: Training a 4D-network in 4D/.

Analysis and plots: An example of running a 4-dimensional already trained network is 4D/AED_analysis.ipynb.

The folder named 4D/ holds training analysis scripts for that amount of dimensions. 

nn_utils.py holds various heplful for networks structures and training functions.
utils holds amongst many, functions for normalization and event filtering.

## Data extraction
The 4-variables in question are:

|Value|
|:---|
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.pt
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.eta
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.phi
HLT_xAOD__JetContainer_TrigHLTJetDSSelectorCollectionAuxDyn.m

## Analysis
fastai saves trained models in the folder models/ relative to the training script, with the .pth file extension. 

In 4D/AED_analysis.py there is analysis of a network with a 3D latent space (i.e. a 4/3 compression ratio), with histogram comparisons of the different values and residual plots. Special attention might be given to these residuals as they tell a lot about the performance of the network.

For a more detailed plots of residuals, see the last part of 4D/AE_analysis.ipynb  

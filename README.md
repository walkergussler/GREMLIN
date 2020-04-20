# Detection Of Recent hcv InfectionS (DORIS)

Python3 software which predicts whether HCV patients have been infected for more or less than one year 

Recently infected patients are often more valuable for peer education and treatment as they are often PWID whom are involved in an ongoing infection process

Previous efforts are either not accurate enough or prone to overtraining deficiencies 

## Feature calculation programs

### final_echlin.py  

Calculates the 11 features used in the final model

### calculate_parameters.py 

Calculates 42 features, some of which are included in the final model. <br/>
Functionality to calculate 68 other experimental features is not contained in this repository

## Feature selection routines

These scripts will take in a CSV file with feature vector information, clusters the feature vectors according to a custom scheme, and then chooses a smaller amount of features according to a relief based feature selection scheme

### kmeans_model_maker.py 

Uses K-means clustering to cluster the features, and then select the feature from each cluster which has the highest model importance score 

### correlation_cliques.py 

Computes a graph over the set of correlations between features, and draws an edge between two features if their correlation is greater than a varying correlation threshold T. Following this, we iterate through the cliques sorted by the positive part of their model importance score according to the relief based function. When we observe each clique, we select the variable with the highest model importance score to participate in the candidate model, while marking all the other features present in the clique 

## Other

### data_normalizer.py 

Takes in a CSV output from a feature calculation program and scales the data linearly to the floating point range [0,1], also adds known recent/chronic labels to the dataset

### wrapper.py

Trains a model based off data in recent and chronic training folders, issues predictions on test folder

```
usage: wrapper.py [-h] [-r RECENT] [-c CHRONIC] [-t TEST] [-o OUTPUT] [-f]
```

optional arguments:

  -h, --help: Show this help message and exit

  -r, --recent: Path to input folder with recent samples
  
  -c, --chronic: Path to input folder with chronic samples
  
  -t, --test: Path to input folder with test samples
  
  -o, --output: Desired output file name
  
  -f, --fullfile: Pass this as an argument to process the whole file rather than the largest 1-step connected component. Shown to be less accurate than only processing the largest one-step connected network.

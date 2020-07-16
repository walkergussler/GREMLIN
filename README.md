#  automagic Genomic featuRe Engineering and Machine LearnINg (GREMLIN)
automated genomic feature engineering, feature selection and machine learning package for data mining and accurate automatic machine learning oh viral populations sequenced via amplicon NGS sequencing. 

Software written in Python3. Originally designed to predict whether HCV patients have been infected for more or less than one year. Recently infected patients are often more valuable for peer education and treatment as they are often PWID whom are involved in an ongoing infection process

Previous efforts are either not accurate enough or prone to overtraining deficiencies. We realized that we would need to be able to obtain new feature sets on the fly whenever we are presented with new data. This package solves that problem by calculating a feature set of roughly 100 biologically relevant variables, then selecting a small subset of the original variables, then performing sequential feature selection to choose a model which is as accurate and generalizable as possible. 

The software cannot distinguish data with randomly assigned labels, however can consistently deliver models with 96% accuracy on the HCV recency problem. 

## Using the software

#### calculate_features.py

Input: a folder with ```.FASTA``` files in which each genomic sample is represented by a multiple sequence alignment (MSA) of the population of variants. Each unique sequence gets its own entry, and frequencies are represented in the ID line following the last underscore. 

Output: Calculates the large feature set, makes a file ```values.csv``` in which the rows are built from the list of genomic samples, and the columns are the engineered features. 

#### automatic_feature_selection.py

Input: ```values.csv``` output file from ```calculate_features.py```. 

Output: ```final_model.csv``` as well as some text to STDOUT which displays accuracy information and time statements.

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

# A Systematic Evaluation of Machine Learning-based Biomarkers for Major Depressive Disorder
This is the accompanying repository for our manuscript titled "A Systematic Evaluation of Machine 
Learning-based Biomarkers for Major Depressive Disorder". 

## Installation
All analyses were run using Python 3.8 on a Linux server. To run the code, create a conda environment and install all
necessary Python packages using the requirements.txt file.

```bash
conda create -n multivariate_biomarker python==3.8
conda activate multivariate_biomarker
pip install -r requirements.txt
```

## Data Preprocessing
All neuroimaging data modalities have been preprocessed as described in the methods section and supplementary materials of the manuscript.
Once the preprocessing was done for all modalities, the data was vectorized to create a n_samples x n_features data matrix.
These matriced and the corresponding diagnostic label was saved as X.npy and y.npy in a dedicated folder for every analysis.
Since the data used in this study cannot be shared publicly, example data was created using sklearn's `make_classification()` 
function. The example data is saved under `data/example_modality/` and is provided with this repository. The script to
create the example data is `make_example_data.py`.

## Machine Learning Analyses
The main machine learning pipelines are implemented using PHOTONAI. There are six dedicated pipelines that cover a wide
range of different machine learning algorithms including random forests, logistic regression, support vector machines,
naive bayes, k-nearest neighbours and boosting. A single PHOTONAI Hyperpipe for every algorithm was used for the analyses.
Note, the six algorithms were trained separately to investigate the upper limit of the classification accuracy that can 
be expected for HC versus MDD classification. For any other analyses, all algorithms can (and should be) added to a single
PHOTONAI Hyperpipe instance. This way, PHOTONAI is able to optimize the choice of the algorithm itself as a hyperparameter
of the complete machine learning pipeline.

An example ML analysis is provided in file `run_example.py`.

## Postprocessing
Once the analyses for all modalities and all subsamples have been run, a postprocessing is done to estimate the effect
of a reliability improvement as well as the multimodal integration of the predictions from the unimodal model. Note that
only the predictions from the test sets are used in the multimodal voting classifier to ensure that no data leakage can
bias the estimate of the generalizability.

## Figures







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
cd scientific_plots
pip install -e .
```

## Data Preprocessing
All neuroimaging data modalities have been preprocessed as described in the methods section and supplementary materials of the manuscript.
Once the preprocessing was done for all modalities, the data was vectorized to create a n_samples x n_features data matrix.
These matriced and the corresponding diagnostic label was saved as X.npy and y.npy in a dedicated folder for every analysis.
Since the data used in this study cannot be shared publicly, example data was created using sklearn's `make_classification()` 
function. The example data is saved under `data/dummy_modality/` and is provided with this repository. The script to
create the dummy data is `01_make_dummy_data.py`.

## Machine Learning Analyses
The main machine learning pipelines are implemented using PHOTONAI. There are six dedicated pipelines that cover a wide
range of different machine learning algorithms including random forests, logistic regression, support vector machines,
naive bayes, k-nearest neighbours and boosting. A single PHOTONAI Hyperpipe for every algorithm was used for the analyses.
Note, the six algorithms were trained separately to investigate the upper limit of the classification accuracy that can 
be expected for HC versus MDD classification. For any other analyses, all algorithms can (and should be) added to a single
PHOTONAI Hyperpipe instance. This way, PHOTONAI is able to optimize the choice of the algorithm itself as a hyperparameter
of the complete machine learning pipeline.

An example ML analysis is provided in file `02_run_example.py`. Once you run this script, all classification pipelines
will be computed and the results are saved to `results/dummy_modality/filter_hc_mdd/pipeline_results`. This analysis
runs approximately 30-60 minutes for this dummy data modality.

Of course, you
can also run this script using any other data that you provide. Just create a folder structure similar to the one
provided in the example dummy_modality. You can then copy the `dummy_modality.yaml` file in the `analyses` folder and change
the name of data modality to whatever your folder and modality is called. 

## Postprocessing
Once the analyses for all modalities and all subsamples have been run, a postprocessing is done to estimate the effect
of a reliability improvement as well as the multimodal integration of the predictions from the unimodal model. Note that
only the predictions from the test sets are used in the multimodal voting classifier to ensure that no data leakage can
bias the estimate of the generalizability.

An example is provided in the file `03_analyze_results.py`. Run this script to collect the results and produce multimodal
integration of single-modality predictions. This script will also produce tables and figures. The aggregated results
are saved to a folder called `aggregated`.

## Reliability Simulation Analysis
The postprocessing script `03_analyze_results.py` also includes the code to run the reliability analysis. In essence,
Matthew's correlation coefficient is calculated based on the model predictions and true diagnostic labels. These
correlation coefficients are then transformed as described in the publication based on classical test theory. The 
reliability corrected correlations are then back-transformed to classification accuracy to compare them with the
original uncorrected results.

## Analysis of Systematic Model Errors
Investigating the systematic predictions errors of a machine learning model can be helpful in uncovering which
patient subgroups are easiest or most difficult to identify. Run the script `04_analysis_of_model_errors.py` to run this 
analysis. This is done for a single modality, which is the dummy modality in this example.

## Variational Autoencoder Neural Networks
As additional non-linear dimensionality reduction method, we investigated the effect of variational autoencoder
neural networks (standard and contrastive) on model performance. Run the script `05_variational_autoencoder.py` to
generate latent embeddings of the data using VAE models. These embeddings can be used to run the previous machine learning
pipeline and evaluate the effect on classification performance.

# Author
If you have any questions on this code or our analyses, please feel free to contact me either by opening a Github Issue
in this repo or by mailing me directly.

Nils Winter
nils.r.winter@uni-muenster.de





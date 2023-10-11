"""
Run this script to create dummy data
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


X, y = make_classification(
    n_samples=1800,
    n_features=150,
    n_informative=20,
    n_redundant=10,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=True
)

dummy_data_folder = os.path.join('results', 'dummy_modality', 'filter_hc_mdd', 'merger_data')
sample_info_folder = os.path.join('results', 'dummy_modality', 'filter_hc_mdd', 'data_information')

# create the necessary folders
os.makedirs(dummy_data_folder, exist_ok=True)
os.makedirs(sample_info_folder, exist_ok=True)

# save data as numpy
np.save(os.path.join(dummy_data_folder, 'X.npy'), X)
np.save(os.path.join(dummy_data_folder, 'y.npy'), y)

# now fake some information on the sample (you will need a subject ID for every subject, the column should be called
# 'Proband')
df = pd.DataFrame({"Proband": np.arange(X.shape[0])})
df.to_csv(os.path.join(sample_info_folder, 'sampled_data.csv'))


"""
Run this script to create dummy data
"""

import os
import numpy as np
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

dummy_data_folder = os.path.join('results', 'dummy_modality', 'filter_hc_mdd')
os.makedirs(dummy_data_folder)
np.save(os.path.join(dummy_data_folder, 'X.npy'), X)
np.save(os.path.join(dummy_data_folder, 'y.npy'), y)

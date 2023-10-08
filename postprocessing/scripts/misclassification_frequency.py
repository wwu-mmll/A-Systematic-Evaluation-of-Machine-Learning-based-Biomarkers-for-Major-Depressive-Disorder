import numpy as np
from tqdm import tqdm
from sklearn.utils import resample

from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter


def bootstrap_run(boot_ind: int, photonai_pipeline, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    """
    Do a single bootstrapped training run using a photonai pipeline
    :param boot_ind: number of bootstrap run
    :param photonai_pipeline: optimum pipe that has previously been trained with photonai
    :param X_train: training data (2d)
    :param y_train: training targets (1d)
    :param X_test: test data (2d)
    :return:
    """
    simplefilter("ignore", category=ConvergenceWarning)
    if boot_ind:
        X_train_boot, y_train_boot = resample(X_train, y_train, random_state=boot_ind)
    else:
        # if boot_ind is 0, don't do any bootstrap resampling
        X_train_boot, y_train_boot = X_train, y_train

    # make sure to get an empty (not already trained) photonai pipeline
    model = photonai_pipeline.copy_me()

    # train on training data
    model.fit(X_train_boot, y_train_boot)

    # return predictions for test set
    return model.predict(X_test)


def bootstrap_misclassification(X: np.ndarray, y: np.ndarray, photonai_pipeline,
                                cv, n_boot: int = 100):
    """

    :param X: complete sample data
    :param y: targets
    :param photonai_pipeline:
    :param cv: sklearn cross validation object
    :param n_boot: number of bootstrap runs to perform
    :return: original predictions without bootstrapping and bootstrapped predictions
    """
    # create empty arrays for predictions
    preds_original = np.empty(X.shape[0])
    preds_boot = np.empty((X.shape[0], n_boot))

    fold_ind = 1
    for train, test in cv.split(X, y):
        print(f"\nRunning Fold {fold_ind}")
        fold_ind += 1
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]

        fold_preds = list()
        # use tqdm to get nice progress bars
        for boot_ind in tqdm(range(n_boot + 1), ascii=True, desc="Running bootstrap"):
            # everytime a bootstrap run is fit, append test set predictions
            fold_preds.append(bootstrap_run(boot_ind, photonai_pipeline, X_train, y_train, X_test))
        folds_preds = np.asarray(fold_preds).transpose()

        # original (no bootstrap) predictions are in the first column, bootstrapped predictions in all other columns
        preds_original[test] = folds_preds[:, 0]
        preds_boot[test] = folds_preds[:, 1:]

    # return predictions in exactly the same order as the original data
    return preds_original, preds_boot


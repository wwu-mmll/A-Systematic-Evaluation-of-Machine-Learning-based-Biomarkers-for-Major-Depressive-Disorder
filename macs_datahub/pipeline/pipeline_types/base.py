import os
import abc
import numpy as np

from joblib import Memory


cache_dir = './tmp/kernel_cache'

memory = Memory(location=cache_dir, verbose=0)


class PipelineBase:

    """ base class for all photonai pipelines

    Attributes
    ----------
    X: nparray
        contains data for photonai
    y: nparray
        contains targets for photonai
    path_to_project: str
        directory to project
    config: dict
        dictionary containing config information set by user
    sample_filter: str
        name of the filter used
    n_processes: int
        number of processes used

    Methods
    -------
    finalize()
        creates a "done.txt" if pipeline calculation is finished
    run()
        checks if "redo" bool is true or "done.txt" does not exist and runs "_run"-function again
    _run()
        abstract method implemented by children classes (pipeline types)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, path_to_project: str,
                 config: dict, sample_filter: str, n_processes: int = 1, is_imbalanced_data: bool = False,
                 is_multi_modal: bool = False):

        self.X = X
        self.y = y
        self.path_to_project = path_to_project
        self.config = config
        self.filter = sample_filter
        self.n_processes = n_processes
        self.is_imbalanced_data = is_imbalanced_data
        self.is_multi_modal = is_multi_modal
        self.pipe = None

    def finalize(self):

        """ creates a "done.txt" if pipeline calculation is finished """

        with open(os.path.join(self.path_to_project, 'done.txt'), 'w') as f:
            f.write('')

    def run(self, redo: bool = False):

        """ checks if "redo" bool is true or "done.txt" does not exist and runs "_run"-function again """

        if redo:
            self._create()
            self._fit()
        else:
            if not os.path.exists(os.path.join(self.path_to_project, 'done.txt')):
                self._create()
                self._fit()
            else:
                print(f"Pipeline already exists in results dir {os.path.join(self.path_to_project)}. Skipping.\n")

    @abc.abstractmethod
    def _fit(self):

        """ abstract method implemented by children classes (pipeline types) """

        raise NotImplementedError("Please implement this method")

    @abc.abstractmethod
    def _create(self):

        """ abstract method implemented by children classes (pipeline types) """

        raise NotImplementedError("Please implement this method")
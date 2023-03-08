import os
import numpy as np
import importlib
import importlib.util

from macs_datahub.config_reader import ConfigReader


def run_pipeline(config_file_dir: str, sample_filter: str = None, pipeline: str = None):

    """ runs pipelines with specified parameters set in config

    Parameters
    ----------
    config_file_dir: str
        directory to config file
    sample_filter: str
        name of sample filter
    pipeline: str
        name of pipeline type
    """

    # read config file
    config = ConfigReader.execute(config_file_dir)

    pipeline_config = config['pipelines']
    if pipeline_config['skip']:
        return

    # reading path configs and create directories
    output_config = config['output_config']
    if output_config['output_dir'] is None:
        raise SyntaxError("Please add output directory to config file using keyword: output_dir")
    else:
        output_dir = output_config['output_dir']

    if output_config['name'] is None:
        raise SyntaxError("Please add project name to config file using keyword: name")
    else:
        output_name = output_config['name']

    if (output_config['sample_filter'] is None) and (sample_filter is None):
        raise SyntaxError("Please add sample filter to config file using keyword: sample_filter"
                          "or specify a single sample filter when calling main.py in the terminal")
    elif sample_filter:
        print(f'A single filter was specified, overwriting sample filters in yaml file.'
              f'Only using {sample_filter}.')
        sample_filter = [sample_filter]
    else:
        sample_filter = output_config['sample_filter']

    if (pipeline_config['types'] is None) and (pipeline is None):
        raise SyntaxError("Please add pipeline type to config file using keyword: types"
                          "or specify a single pipeline when calling main.py in the terminal")
    elif pipeline:
        pipeline_config['types'] = [pipeline]
    else:
        pass

    if 'n_processes' not in pipeline_config.keys():
        pipeline_config['n_processes'] = 1

    if 'redo' not in pipeline_config.keys():
        pipeline_config['redo'] = False

    if 'is_multi_modal' not in pipeline_config.keys():
        is_multi_modal = False
    else:
        is_multi_modal = pipeline_config['is_multi_modal']

    if pipeline_config['is_imbalanced_data'] is None:
        raise SyntaxError("Please add output directory to config file using keyword: output_dir")
    else:
        is_imbalanced_data = pipeline_config['is_imbalanced_data']

    for single_filter in sample_filter:

        # set output path
        path_to_project = os.path.join(output_dir, output_name, single_filter)
        results_dir = os.path.join(path_to_project, 'pipeline_results')
        os.makedirs(results_dir, exist_ok=True)

        # get data
        path_nparrays = os.path.join(path_to_project, "merger_data")
        X = np.load(os.path.join(path_nparrays, "X.npy"))
        y = np.load(os.path.join(path_nparrays, "y.npy"))

        # replace targets with 0s and 1s
        y_classes = list(set(y))
        is_class_1 = (y == y_classes[0])
        is_class_2 = (y == y_classes[1])
        y[is_class_1] = 0
        y[is_class_2] = 1

        # run pipelines
        for pipeline_type in pipeline_config['types']:
            desired_class_home = 'pipeline.pipeline_types'
            desired_class_name = pipeline_type
            imported_module = importlib.import_module(desired_class_home)
            desired_class = getattr(imported_module, desired_class_name)
            pipeline_class = desired_class(X=X, y=y, path_to_project=os.path.join(results_dir, pipeline_type),
                                           config=config, sample_filter=single_filter,
                                           n_processes=pipeline_config['n_processes'],
                                           is_imbalanced_data=is_imbalanced_data,
                                           is_multi_modal=is_multi_modal)

            pipeline_class.run(pipeline_config['redo'])
            pipeline_class.finalize()

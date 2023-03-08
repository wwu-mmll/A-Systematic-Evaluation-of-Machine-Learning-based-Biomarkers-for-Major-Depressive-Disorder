import os
import json
import argparse
import pandas as pd

from glob import glob
from datetime import datetime
from photonai.processing import ResultsHandler
from macs_datahub.pipeline import run_pipeline
from macs_datahub.config_reader import ConfigReader
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def main():

    """ main function to run the preprocessing and the photonai-pipelines for the multi modality project. """

    # read args set by user
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_folder", help="Specify folder where all yaml files are located",
                        default=None)
    parser.add_argument("--yaml_file", help="Specify single yaml configuration file",
                        default=None)
    parser.add_argument("--sample_filter", help="Specify a single filter that should be used in this analysis. "
                                                "This will overwrite the filters defined in the yaml configuration file.",
                        default=None)
    parser.add_argument("--pipeline", help="Specify the PHOTONAI pipeline"
                                            "This will overwrite the pipeline types defined in the yaml configuration file.",
                        default=None)

    args = parser.parse_args()
    dir = args.yaml_folder
    file = args.yaml_file
    sample_filter = args.sample_filter
    pipeline_type = args.pipeline

    if dir:
        print(f'Using {dir} as yaml folder.')
        files = glob(os.path.join(dir, '*.yaml'))
    elif file:
        print(f'Using {file} as yaml file.')
        files = [file]
    else:
        raise NotImplementedError('Specify either yaml_folder or single yaml_file.')

    # run data_creator and pipeline with set config by user for every yaml file present
    df = pd.DataFrame()  # create empty dataframe which is filled by run_result_collector-function
    output_dir = ""  # set empty output directory which is overwritten by run_result_collector-function
    for file in files:

        # core functions to create data and run pipelines; all settings are given in the config-file(s)
        run_pipeline(config_file_dir=file, sample_filter=sample_filter, pipeline=pipeline_type)

        # read config file; has to be executed here once for the information if pipelines are "skipped" or
        # if result collector is run
        config = ConfigReader.execute(file)
        pipeline_config = config['pipelines']
        if not pipeline_config['skip']:
            df, output_dir = run_result_collector(df, config)

    # save collected results and run sanity check on given data by data_creator
    df.to_csv(os.path.join(output_dir, "collected_results.csv"), index=False)


def find_latest_photonai_results(current_results_folder: str):

    try:
        results_folder_photonai = os.listdir(current_results_folder)
    except FileNotFoundError:
        return

    # checks if calculation of pipeline_type is finished by looking at the "done.txt"-file;
    # skips collection of results if "done.txt" does not exist
    if "done.txt" not in results_folder_photonai:
        return "unfinished"
    else:
        results_folder_photonai.remove("done.txt")

    # removes cache for upcoming purposes; does not interfere with running calculations since loop
    # is continued above if "done.txt" is not found.
    if "cache" in results_folder_photonai:
        results_folder_photonai.remove("cache")

    # find latest calculation of pipeline type
    dates = [datetime.strptime(name[-19:], '%Y-%m-%d_%H-%M-%S') for name in results_folder_photonai]
    latest_date = max(dates)

    current_photonai_folder = None
    for tmp_folder in results_folder_photonai:
        if latest_date.strftime('%Y-%m-%d_%H-%M-%S') in tmp_folder:
            current_photonai_folder = tmp_folder

    return current_photonai_folder


def run_result_collector(df, config):

    """ collects the result of the calculations defined in the config.

    Parameters
    ----------
    df: DataFrame
        dataframe of summarized results; appended by the result collector itself
    config: dict
        dictionary containing config information set by user
    """

    # reading path configs and create directories
    output_config = config['output_config']
    output_dir = output_config['output_dir']
    sample_filter = output_config['sample_filter']
    calc_name = output_config['name']

    column_list = ["modality", "sample_filter"]

    # first step in order to get to the results is to create the string of the directories leading to the results.
    #
    # The directories have the following structure:
    # 1) output directory set by user in config - static for every yaml file
    # 2) calculation name set by user in config - static for every yaml file
    # 3) filter name set by user in config - several filters can be set per yaml file
    # 4) a folder always named "pipeline_results" - static
    # 5) pipeline name set by user in config - several pipelines can be set per yaml file
    #
    # The run_result_collector-function gets called for every yaml file in the main function
    for filter_name in sample_filter:

        results_folder_filter = os.path.join(output_dir, calc_name, filter_name, "pipeline_results")
        results_folder_pipeline_types = glob(os.path.join(results_folder_filter, "*"))

        combined_results = list()
        for current_results_folder in results_folder_pipeline_types:

            pipe_type = os.path.basename(current_results_folder)

            if not pipe_type in column_list:
                column_list.append(str(pipe_type))

            latest_photonai_folder = find_latest_photonai_results(current_results_folder)
            if not latest_photonai_folder:
                continue

            # load json containing results
            result_json = os.path.join(current_results_folder, latest_photonai_folder, "photon_result_file.json")
            result_data = json.load(open(result_json))

            handler = ResultsHandler()
            handler.load_from_file(result_json)

            # get results in json
            bacc_mean = result_data["metrics_test"][0]["value"]
            metric_name = result_data["metrics_test"][0]['metric_name']
            bacc_std = result_data["metrics_test"][1]["value"]

            current_results = {'AnalysisName': calc_name,
                               'Sample': filter_name,
                               'PipelineType': pipe_type,
                               'Metric': metric_name,
                               'Mean': bacc_mean,
                               'Std': bacc_std,
                               'Folds': handler.get_performance_outer_folds()[metric_name]}
            combined_results.append(current_results)

        # transform results to dataframe (df2) and concatenate to summarizing dataframe (df)
        df2 = pd.DataFrame(combined_results)
        df = pd.concat([df, df2])

    df.dropna(axis=0, how='all', inplace=True)
    return df, output_dir


if __name__ == "__main__":
    main()

import os
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from glob import glob
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, recall_score, f1_score, roc_auc_score, \
    accuracy_score

from macs_datahub.main import find_latest_photonai_results
from photonai.processing import ResultsHandler

#from scientific_plots.plots.corr_plot import corr_plot
#from scientific_plots.utils import savefig


def calculate_metrics(frame):
    # ToDo: "-1" only makes sense for HC MDD analysis
    if 'Group' in frame.keys():
        y_true = frame['Group'] - 1
    else:
        y_true = frame['y_true']
    y_pred = frame['y_pred']
    if np.isnan(frame['probabilities'].iloc[0]).any():
        y_score = frame['y_pred']
    elif isinstance(frame['probabilities'].iloc[0], float):
        y_score = frame['probabilities']
    else:
        # frame['probabilities'] contains a list of objects that look like this [0.55, 0.45], looks like an array
        # but it is some kind of weird object and you cannot simply make it an array using np.asarray
        # that's why I loop through it and select all these objects through pandas indexing, somehow this makes it
        # a proper array, the resulting list can be converted using np.asarray
        y_score = np.asarray([frame['probabilities'].iloc[i] for i in range(len(frame['probabilities']))])[:, 1]
    return pd.DataFrame({'BACC': [balanced_accuracy_score(y_true=y_true, y_pred=y_pred)],
                         'Sensitivity': [recall_score(y_true=y_true, y_pred=y_pred, pos_label=1)],
                         'Specificity': [recall_score(y_true=y_true, y_pred=y_pred, pos_label=0)],
                         'AUC': [roc_auc_score(y_true=y_true, y_score=y_score)],
                         'MCC': [matthews_corrcoef(y_true=y_true, y_pred=y_pred)],
                         'f1_score': [f1_score(y_true=y_true, y_pred=y_pred)],
                         'ACC': [accuracy_score(y_true=y_true, y_pred=y_pred)]})


def calculate_prevalence(df):
    return np.sum(df['y_true']) / df.shape[0]


def calculate_bias(df):
    return np.sum(df['y_pred']) / df.shape[0]


def correlate(df):
    return matthews_corrcoef(df['y_true'], df['y_pred'])


def calculate_bacc(df):
    return balanced_accuracy_score(df['y_true'], df['y_pred'])


def mcc_to_bacc(df):
    r = df['MCC_corrected_empirical_pheno'].to_numpy()
    prevalence = df['prevalence'].to_numpy()
    bias = df['bias'].to_numpy()
    return np.asarray(1 / (2 * np.sqrt((prevalence - np.square(prevalence)) / (bias - np.square(bias)))) * r + 0.5)[0]


def mcc_to_bacc_v2(df, var_name):
    r = df[var_name].to_numpy()
    prevalence = df['prevalence'].to_numpy()
    bias = df['bias'].to_numpy()
    return np.asarray(1 / (2 * np.sqrt((prevalence - np.square(prevalence)) / (bias - np.square(bias)))) * r + 0.5)


def calculate_mean_sd_over_folds(df, groupby: list = None):
    if groupby:
        df_mean = df.groupby(groupby).mean()
        df_sd = df.groupby(groupby).std()
    else:
        df_mean = pd.DataFrame(df.mean()).T
        df_sd = pd.DataFrame(df.std()).T
    df_sd = df_sd.add_suffix('_sd')
    df_mean_sd = pd.concat([df_mean, df_sd], axis=1)
    df_mean_sd = df_mean_sd.reset_index()
    return df_mean_sd


def aggregate_results(results_folder: str, filter_names: list, pipeline_types: list, data_folder: str):
    """
    Load results from all modalities, collect classification metrics and calculate matthew's correlation and auroc
        calculate mean and standard deviation for all metrics
        also load predictions and save everything neat and clean to csv files
    :param results_folder:
    :param filter_names:
    :param pipeline_types:
    :param data_folder:
    :return:
        Does not return anything but results are saved to 'aggregated' folder
    """

    # initialize empty dataframe
    df = pd.DataFrame()
    df_mean = pd.DataFrame()
    df_config = pd.DataFrame()
    df_predictions = pd.DataFrame()

    # find all available modalities
    modalities = glob(os.path.join(results_folder, "*/"))

    for mod in modalities:
        for filter_name in filter_names:
            for pipeline_type in pipeline_types:
                print("Loading results for {} {} {}".format(mod, filter_name, pipeline_type))
                current_result_folder = os.path.join(mod, filter_name, "pipeline_results",
                                                     pipeline_type)

                latest_photonai_folder = find_latest_photonai_results(current_result_folder)
                if not latest_photonai_folder:
                    continue

                # load json containing results
                result_json = os.path.join(current_result_folder, latest_photonai_folder, "photonai_results.json")
                current_config = {'Modality': [os.path.basename(os.path.normpath(mod))],
                                  'Sample': [filter_name],
                                  'PipelineType': [pipeline_type],
                                  'AnalysisName': [None],
                                  'ModalityIntegration': [None]}
                try:
                    if latest_photonai_folder == "unfinished":
                        raise ValueError("PHOTONAI pipeline not finished")
                    result_data = json.load(open(result_json))

                    handler = ResultsHandler()
                    handler.load_from_file(result_json)

                    # get test set predictions
                    test_pred = pd.DataFrame(handler.get_test_predictions())

                    # load information on subjects from data folder and make correspondance between original ID and pred
                    mod_name = os.path.basename(os.path.normpath(mod))
                    cpath_data = os.path.join(data_folder, mod_name, filter_name, 'data_information')
                    data_info = pd.read_csv(os.path.join(cpath_data, "sampled_data.csv"), low_memory=False)
                    nsubs = test_pred.shape[0]
                    if mod_name == 'multi_modality':
                        mod_integration = 'PCA-based'
                        analysis_name = 'Data Integration'
                    else:
                        mod_integration = 'None'
                        analysis_name = mod_name

                    cpreds = pd.DataFrame({'Proband': data_info['Proband'],
                                           'y_true': test_pred['y_true'],
                                           'y_pred': test_pred['y_pred'],
                                           'y_score': test_pred['probabilities'],
                                           'fold_id': test_pred['fold'],
                                           'Modality': [mod_name] * nsubs,
                                           'Sample': [filter_name] * nsubs,
                                           'PipelineType': [pipeline_type] * nsubs,
                                           'AnalysisName': [analysis_name] * nsubs,
                                           'ModalityIntegration': [mod_integration] * nsubs})
                    current_config['AnalysisName'] = [analysis_name]
                    current_config['ModalityIntegration'] = [mod_integration]

                    # get metrics
                    folds = handler.get_performance_outer_folds()
                    bacc = folds['balanced_accuracy']
                    acc = folds['accuracy']
                    sens = folds['sensitivity']
                    spec = folds['specificity']
                    f1 = folds['f1_score']
                    # calculate additional metrics
                    additional_metrics = test_pred.groupby(by=['fold']).apply(calculate_metrics).reset_index()
                    auc = additional_metrics['AUC'].tolist()
                    mcc = additional_metrics['MCC'].tolist()

                    # get number of samples per group
                    _, cnts = np.unique(test_pred['y_true'], return_counts=True)

                    current_config["Status"] = [1]
                    current_config_df = pd.DataFrame(current_config)
                    df_config = pd.concat([df_config, current_config_df])
                except BaseException as e:
                    print(e)
                    current_config["Status"] = [0]
                    current_config_df = pd.DataFrame(current_config)
                    df_config = pd.concat([df_config, current_config_df])
                    continue

                n_folds = len(bacc)
                current_results = {'Modality': [mod_name] * n_folds,
                                   'Sample': [filter_name] * n_folds,
                                   'PipelineType': [pipeline_type] * n_folds,
                                   'AnalysisName': [analysis_name] * n_folds,
                                   'ModalityIntegration': [mod_integration] * n_folds,
                                   'BACC': bacc,
                                   'ACC': acc,
                                   'Sensitivity': sens,
                                   'Specificity': spec,
                                   'AUC': auc,
                                   'f1_score': f1,
                                   'MCC': mcc}

                # transform results to dataframe (df2) and concatenate to summarizing dataframe (df)
                df_current = pd.DataFrame(current_results)
                df_current_mean_sd = calculate_mean_sd_over_folds(df_current,
                                                                  groupby=['Modality', "PipelineType", "Sample",
                                                                           'AnalysisName', 'ModalityIntegration'])

                df_current_mean_sd['nHC'] = [cnts[0]]
                df_current_mean_sd['nMDD'] = [cnts[1]]

                df = pd.concat([df, df_current])
                df_mean = pd.concat([df_mean, df_current_mean_sd])
                # concatenate prediction dataframes
                df_predictions = pd.concat([df_predictions, cpreds])

    mods = {"freesurfer": "Freesurfer",
            "cat12": "VBM",
            "hariri": "Hariri",
            "resting_state": "RS Connectivity",
            "graph_metrics_rs": "RS Network Parameters",
            "alff": "ALFF",
            "falff": "fALFF",
            "lcor": "LCOR",
            "dti_fa": "FA",
            "dti_md": "MD",
            "graph_metrics_dti": "DTI Network Parameters",
            "prs": "PRS",
            "childhood_maltreatment": "Childhood Maltreatment",
            "social_support": "Social Support",
            "multi_modality": "all"
            }
    pipes = {'BoostingPipeline': 'Boosting Classifier',
             'KNNPipeline': 'k-Nearest Neighbours',
             'LogisticRegressionPipeline': 'Logistic Regression',
             'NaiveBayesPipeline': 'Gaussian Naive Bayes',
             'RandomForestPipeline': 'Random Forest',
             'SVMPipeline': 'Support Vector Machine',
             'UnivariateLogisticRegressionPipeline': 'Logistic Regression'}

    for current_df in [df_predictions, df, df_mean, df_config]:
        current_df['Modality'] = current_df['Modality'].replace(mods)
        current_df['PipelineType'] = current_df['PipelineType'].replace(pipes)
        current_df['AnalysisName'] = current_df['AnalysisName'].replace(mods)

    os.makedirs("./aggregated", exist_ok=True)
    df_predictions.to_csv('aggregated/predictions.csv')
    df.to_csv('aggregated/results.csv')
    df_mean.to_csv('aggregated/results_mean_sd.csv')
    df_config.to_csv('aggregated/analyses_overview.csv')
    return


def create_ensemble_preds(file: str = 'aggregated/predictions.csv', samples: list = None,
                          pipelines: list = None, modalities: list = None,
                          add_to_results: bool = False):
    """

    :param file:
    :param samples:
    :param pipelines:
    :param modalities:
    :param add_to_results:
    :return:
    """
    if modalities is None:
        modalities = ['Freesurfer', 'VBM', 'Hariri', 'RS Connectivity', 'ALFF', 'fALFF', 'LCOR',
                      'FA', 'MD', 'DTI Network Parameters', 'RS Network Parameters']
    if pipelines is None:
        pipelines = ["Boosting Classifier", "k-Nearest Neighbours", "Logistic Regression",
                     "Gaussian Naive Bayes", "Random Forest", "Support Vector Machine"]
    if samples is None:
        samples = ['filter_hc_mdd', 'filter_hc_mdd_acute', 'filter_hc_mdd_severe']

    original_df = pd.read_csv(file)

    ensemble_metrics_all = pd.DataFrame()
    ensemble_metrics_agg_all = pd.DataFrame()
    sum_over_both_all = pd.DataFrame()

    for sample in samples:
        df = original_df.copy()
        df = df[df['Sample'] == sample]
        df = df[df['PipelineType'].isin(pipelines)]

        # get dataframe and results for pca-based multi modality integration analysis
        # we will need the fold_id for this sample to later apply the same cross-validation splits when calculating
        # the ensemble metrics
        # use one algorithm (e.g. Logistic Regression) just to get the cv (it's the same for all algorithms)
        df_pca_based_modality_integration = df[df['Modality'].isin(['all']) &
                                               df['PipelineType'].isin(['Logistic Regression'])]
        df_pca_based_modality_integration = df_pca_based_modality_integration[['Proband', 'fold_id']]
        df_pca_based_modality_integration.set_index('Proband', inplace=True)

        df = df[df['Modality'].isin(modalities)]

        # select only subjects with all modalities
        n_analyses_per_subject = df.groupby(by='Proband').count().reset_index()
        n_max = np.max(n_analyses_per_subject['y_pred'])
        subs_complete = n_analyses_per_subject['Proband'][n_analyses_per_subject['y_pred'] == n_max]
        df = df[df['Proband'].isin(subs_complete.tolist())]

        # count number of pipelines and modalities per subject
        n_pipes_per_subject = df.groupby(by=['Proband', 'Modality']).count().reset_index()
        n_modalities_per_subject = df.groupby(by=['Proband', 'PipelineType']).count().reset_index()

        # calculate sum across modalities or pipelines or both (sum should work since 0=HC and 1=MDD)
        ensemble_over_pipelines = df.groupby(by=['Proband', 'Modality']).agg({'y_pred': 'sum',
                                                                              'y_true': 'first',
                                                                              'Sample': 'first',
                                                                              'PipelineType': 'first',
                                                                              'y_score': 'first'}).reset_index()

        ensemble_over_modalities = df.groupby(by=['Proband', 'PipelineType']).agg({'y_pred': 'sum',
                                                                                   'y_true': 'first',
                                                                                   'Sample': 'first',
                                                                                   'Modality': 'first',
                                                                                   'y_score': 'first'}).reset_index()
        ensemble_over_both = df.groupby(by=['Proband']).agg({'y_pred': 'sum',
                                                             'y_true': 'first',
                                                             'Sample': 'first',
                                                             'PipelineType': 'first',
                                                             'Modality': 'first',
                                                             'y_score': 'first'}).reset_index()

        corr = df.pivot(index=['Proband', 'PipelineType'], columns='AnalysisName',
                 values='y_pred').corr()
        corr = corr[modalities].transpose()[modalities]

        cm_no_ones = df.pivot(index=['Proband', 'PipelineType'], columns='AnalysisName',
                 values='y_pred').groupby(by='PipelineType').corr()
        cm_no_ones[cm_no_ones == 1] = 0
        print(f"min = {cm_no_ones.to_numpy().min()} max = {cm_no_ones.to_numpy().max()}")

        corr.to_csv(f'aggregated/correlations_of_predictions_{sample}.csv')
        corr_plot(corr,  vmax=0.5, fig_size=(8 / 2.54, 8 / 2.54))
        ax = plt.gca()
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        plt.tight_layout()
        savefig(f"plots/model_error/corr_plot_all_{sample}.png")

        for pipeline in pipelines:
            corr_matrix = df.pivot(index=['Proband', 'PipelineType'], columns='AnalysisName',
                                   values='y_pred').groupby(by=['PipelineType']).corr().reset_index()
            current_corr_matrix = corr_matrix[corr_matrix['PipelineType'] == pipeline]
            current_corr_matrix = current_corr_matrix.set_index('AnalysisName')
            current_corr_matrix = current_corr_matrix[modalities].transpose()[modalities]

            current_corr_matrix.to_csv(f'aggregated/correlations_of_predictions_from_{sample}_{pipeline}.csv')
            corr_plot(current_corr_matrix, vmax=0.5, fig_size=(8 / 2.54, 8 / 2.54))
            ax = plt.gca()
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            plt.tight_layout()
            plt.suptitle(f'{pipeline}')
            savefig(f"plots/model_error/corr_plot_{sample}_{pipeline}.png")

        # convert sum to something like a probability by dividing by the number of pipelines/modalities/both
        ensemble_over_both['probabilities'] = ensemble_over_both['y_pred'] / n_max
        ensemble_over_pipelines['probabilities'] = ensemble_over_pipelines['y_pred'] / n_pipes_per_subject[
            'PipelineType']
        ensemble_over_modalities['probabilities'] = ensemble_over_modalities['y_pred'] / n_modalities_per_subject[
            'Modality']

        # binarize probability to calculate a single classification label
        ensemble_over_both['y_pred'] = np.asarray(ensemble_over_both['probabilities'] > 0.5).astype(int)
        ensemble_over_pipelines['y_pred'] = np.asarray(ensemble_over_pipelines['probabilities'] > 0.5).astype(int)
        ensemble_over_modalities['y_pred'] = np.asarray(ensemble_over_modalities['probabilities'] > 0.5).astype(int)

        # merge with pca based results
        def merge_with_pca_based_results(ensemble_df, df_pca_based):
            ensemble_df.set_index('Proband', inplace=True)
            ensemble_df = ensemble_df.join(df_pca_based, how='left')
            return ensemble_df

        ensemble_over_both = merge_with_pca_based_results(ensemble_over_both, df_pca_based_modality_integration)
        ensemble_over_modalities = merge_with_pca_based_results(ensemble_over_modalities,
                                                                df_pca_based_modality_integration)
        ensemble_over_pipelines = merge_with_pca_based_results(ensemble_over_pipelines,
                                                               df_pca_based_modality_integration)

        def aggregate_metrics(df, groupby, sample, modality, pipeline_type, modality_integration, analysis_name):
            df_metrics = df.groupby(by=['fold_id'] + groupby).apply(calculate_metrics).reset_index()
            df_metrics_agg = calculate_mean_sd_over_folds(df_metrics, groupby=groupby)

            for current_df in [df, df_metrics, df_metrics_agg]:
                len_df = current_df.shape[0]
                current_df['Sample'] = [sample] * len_df
                if modality:
                    current_df['Modality'] = [modality] * len_df
                if pipeline_type:
                    current_df['PipelineType'] = [pipeline_type] * len_df
                current_df['AnalysisName'] = [analysis_name] * len_df
                current_df['ModalityIntegration'] = [modality_integration] * len_df
            return df, df_metrics, df_metrics_agg

        # calculate metrics across cv folds (folds are defined by pca-based modality integration analysis)
        ensemble_over_both, metrics_ensemble_both, metrics_ensemble_both_agg = aggregate_metrics(
            df=ensemble_over_both,
            groupby=[],
            sample=sample,
            modality='all',
            pipeline_type='all',
            modality_integration='Voting',
            analysis_name='Ensemble')
        ensemble_over_modalities, metrics_ensemble_modalities, metrics_ensemble_modalities_agg = aggregate_metrics(
            df=ensemble_over_modalities,
            groupby=['PipelineType'],
            sample=sample,
            modality='all',
            pipeline_type=None,
            modality_integration='Voting',
            analysis_name='Modality Ensemble')
        ensemble_over_pipelines, metrics_ensemble_pipelines, metrics_ensemble_pipelines_agg = aggregate_metrics(
            df=ensemble_over_pipelines,
            groupby=['Modality'],
            sample=sample,
            modality=None,
            pipeline_type='all',
            modality_integration='Voting',
            analysis_name='Algorithm Ensemble')

        # concatenate across samples
        ensemble_metrics_all = pd.concat([ensemble_metrics_all, metrics_ensemble_both, metrics_ensemble_modalities,
                                          metrics_ensemble_pipelines])
        ensemble_metrics_agg_all = pd.concat([ensemble_metrics_agg_all, metrics_ensemble_both_agg,
                                              metrics_ensemble_modalities_agg, metrics_ensemble_pipelines_agg])

        sum_over_both_all = pd.concat([sum_over_both_all, ensemble_over_both, ensemble_over_modalities,
                                       ensemble_over_pipelines])

    # write to csv
    ensemble_metrics_all.to_csv('aggregated/results_ensemble.csv')
    ensemble_metrics_agg_all.to_csv('aggregated/results_ensemble_mean_sd.csv')
    sum_over_both_all.to_csv('aggregated/predictions_ensemble.csv')

    # add to existing results file
    if add_to_results:
        for file, ensemble_df in [('aggregated/results.csv', ensemble_metrics_all),
                                  ('aggregated/results_mean_sd.csv', ensemble_metrics_agg_all)]:
            existing_results = pd.read_csv(file)

            # add number of samples to ensemble results
            if 'nHC' in existing_results.keys():
                for sample in samples:
                    nHC = existing_results.loc[(existing_results['Sample'] == sample) & (existing_results['Modality'] == 'all')]['nHC'].iloc[0]
                    nMDD = existing_results.loc[(existing_results['Sample'] == sample) & (existing_results['Modality'] == 'all')]['nMDD'].iloc[0]

                    ensemble_df.loc[ensemble_df['Sample'] == sample, 'nHC'] = [str(int(nHC))] * ensemble_df.loc[ensemble_df['Sample'] == sample].shape[0]
                    ensemble_df.loc[ensemble_df['Sample'] == sample, 'nMDD'] = [str(int(nMDD))] * ensemble_df.loc[ensemble_df['Sample'] == sample].shape[0]

            existing_results = pd.concat([existing_results, ensemble_df], join='inner')
            existing_results.to_csv(os.path.splitext(file)[0] + "_update.csv")
    return


def reliability_corrected_correlations(predictions,
                                       save_to_file: str = 'aggregated/correlations.csv'):
    """
    Calculate corrected correlations and balanced accuracies based on attenuation correction
    :param predictions_file:
    :param save_to_file:
    :return:
    """
    meta_data = pd.read_csv('raw_data/'
                            'modality_reference_data/Clinical_Data.csv',
                            sep=';', na_values=[-99])
    predictions = pd.merge(predictions, meta_data[['Proband', 'Group']], how='left', on='Proband')

    corr_df_all = pd.DataFrame()

    for sample in np.unique(predictions['Sample']):
        predictions_sample = predictions[predictions['Sample'] == sample]

        # calculate MCC for all modalities, pipelines, and also individual folds
        corr_df = predictions_sample.groupby(by=['fold_id', 'Modality', 'PipelineType',
                                                 'AnalysisName', 'ModalityIntegration']).apply(correlate).reset_index()

        corr_df = corr_df.rename({0: 'MCC'}, axis=1)

        # reliability as reported in DSM5 field trials Regier 2013
        reliability = np.ones(10) * 0.28

        # I used this method to replace all fold ids with the corresponding reliability value
        # https://stackoverflow.com/questions/29407945/find-and-replace-multiple-values-in-python
        reliability_vector = corr_df['fold_id'].to_numpy()
        fold_values = np.arange(1, 11)
        tmp_dict = dict(zip(fold_values, reliability))
        reliability_vector = [tmp_dict.get(e, e) for e in reliability_vector]
        corr_df['empirical_reliability'] = reliability_vector

        # correct MCCs for empirical minimum reliability (phenotype)
        corr_df['MCC_corrected_empirical_pheno'] = corr_df['MCC'] / np.sqrt(corr_df['empirical_reliability'])
        corr_df['prevalence'] = predictions_sample.groupby(by=['fold_id', 'Modality', 'PipelineType', 'AnalysisName',
                                                               'ModalityIntegration']).apply(
            calculate_prevalence).to_numpy()
        corr_df['bias'] = predictions_sample.groupby(by=['fold_id', 'Modality', 'PipelineType', 'AnalysisName',
                                                         'ModalityIntegration']).apply(
            calculate_bias).to_numpy()
        corr_df['BACC_corrected_empirical_pheno'] = corr_df.groupby(by=['fold_id', 'Modality', 'PipelineType',
                                                                        'AnalysisName', 'ModalityIntegration']).apply(
            mcc_to_bacc).to_numpy()
        corr_df['BACC'] = predictions_sample.groupby(by=['fold_id', 'Modality', 'PipelineType', 'AnalysisName',
                                                         'ModalityIntegration']).apply(
            calculate_bacc).to_numpy()
        corr_df['Sample'] = [sample] * corr_df.shape[0]

        for rel in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            corr_df[f'MCC_corrected_simulated_{rel}'] = corr_df['MCC'] / np.sqrt(rel)
            corr_df[f'BACC_corrected_simulated_{rel}'] = mcc_to_bacc_v2(corr_df, f'MCC_corrected_simulated_{rel}')
            corr_df[f'MCC_corrected_empirical_and_simulated_{rel}'] = corr_df['MCC'] / np.sqrt(
                rel * np.asarray(reliability_vector))
            corr_df[f'BACC_corrected_empirical_and_simulated_{rel}'] = mcc_to_bacc_v2(corr_df,
                                                                                      f'MCC_corrected_empirical_and_simulated_{rel}')

        corr_df_all = pd.concat([corr_df_all, corr_df])

    corr_df_all.to_csv(save_to_file, float_format='%.3f')
    mean_corr_df = calculate_mean_sd_over_folds(corr_df_all, groupby=['Sample', 'Modality', 'PipelineType',
                                                                      'AnalysisName', 'ModalityIntegration'])
    mean_corr_df.drop(['fold_id', 'fold_id_sd'], axis=1)
    mean_corr_df.to_csv(os.path.splitext(save_to_file)[0] + '_mean_sd.csv', float_format='%.3f')


def create_descriptives_statistics_table(save_to_file, sample_filter: str = None):
    df = pd.read_csv('Clinical_Data.csv', sep=';', na_values=-99.)

    # fix erroneous variables
    df['DurHosp'] = df['DurHosp'].str.replace(',', '.').astype(float)
    df['DurDep'] = df['DurDep'].str.replace(',', '.').astype(float)
    df['FSozU_Sum'] = df['FSozU_Sum'].str.replace(',', '.').astype(float)

    # select specific sample
    if sample_filter:
        df, _ = SampleFilter().apply_filter(sample_filter, df)
        df['Group'] = df['Group'].replace({'HC': 'Healthy', 'MDD': 'Major Depression'})
    else:
        df['Group'] = df['Group'].replace({1: 'Healthy', 2: 'Major Depression'})

    # rename variables to proper names for publication
    col_names = {'Alter': 'Age',
                 'Geschlecht': 'Sex',
                 'HAMD_Sum21': 'HAMD',
                 'BDI_Sum': 'BDI',
                 'CTQ_Sum': 'CTQ',
                 'FSozU_Sum': 'Social Support',
                 'Sum_MED': 'Medication Index',
                 'Hosp': 'Number of previous inpatient treatments',
                 'DepEp': 'Number of previous depressive episodes',
                 'DurHosp': 'Total duration of previous inpatient treatments '
                            '(in weeks)',
                 'DurDep': 'Total duration of all previous depressive episodes '
                           '(in months)'}
    df = df.rename(col_names, axis=1)

    df['Sex'] = df['Sex'].replace({1: 'Male', 2: 'Female'})

    # create formatted table with descriptive statistics
    table = descriptive_sample_statistics(df,
                                          title="Social demographics and clinical characteristics of all participants",
                                          footnote="HAMD=Hamilton Rating Scale for Depression. "
                                                   "BDI=Beck Depression Inventory. "
                                                   "CTQ=Childhood Trauma Questionnaire. "
                                                   "MRI=Magnetic Resonance Imaging. "
                                                   "VBM=Voxel-Based Morphometry. *t or χ² tests.",
                                          groups=['Healthy', 'Major Depression'],
                                          continuous_variables=['Age', 'HAMD', 'BDI', 'CTQ',
                                                                'Social Support', 'Medication Index',
                                                                'Number of previous inpatient treatments',
                                                                'Number of previous depressive episodes',
                                                                'Total duration of previous inpatient treatments '
                                                                '(in weeks)',
                                                                'Total duration of all previous depressive episodes '
                                                                '(in months)',
                                                                ],
                                          categorical_variables=['Sex'],
                                          lancet_format=True)
    table.to_excel(save_to_file, header=False, index=False)
    return


def create_results_table(file: str, sample):
    df = pd.read_csv(file)
    df = df.drop("Unnamed: 0", axis=1)
    df = df[df['Sample'] == sample]

    percentage_metrics = ['BACC', 'ACC', "Sensitivity", 'Specificity', 'AUC']
    percentage_metrics_sd = ['BACC_sd', 'ACC_sd', "Sensitivity_sd", 'Specificity_sd', 'AUC_sd']
    df[percentage_metrics] *= 100
    df[percentage_metrics_sd] *= 100
    df[percentage_metrics] = df[percentage_metrics].round(decimals=1)
    df[percentage_metrics_sd] = df[percentage_metrics_sd].round(decimals=1)
    df[['MCC']] = df[['MCC']].round(decimals=2)
    df[['MCC_sd']] = df[['MCC_sd']].round(decimals=2)
    df = df.astype(str)

    tables_dict = dict()
    tables_dict['Modality'] = df['Modality']
    tables_dict['Algorithm'] = df['PipelineType']

    for metric in percentage_metrics:
        tables_dict[metric] = df[metric] + '% (' + df[metric + '_sd'] + '%)'
    tables_dict['MCC'] = df['MCC'] + ' (' + df['MCC_sd'] + ')'

    tables_dict['n (HC/MDD)'] = df['nHC'] + '/' + df['nMDD']
    tables_dict['Modality Integration'] = df['ModalityIntegration']

    table = pd.DataFrame(tables_dict)
    table = table[table['Modality Integration'] == 'None']
    table.drop(columns='Modality Integration', inplace=True)
    table = table.set_index(['Modality', 'Algorithm'])

    structural = ['Freesurfer', 'VBM', 'FA', 'MD', 'DTI Network Parameters']
    functional = ['Hariri', 'RS Connectivity', 'ALFF', 'fALFF', 'LCOR', 'RS Network Parameters']
    environment_genetics = ['PRS', 'Social Support', 'Childhood Maltreatment']

    mod_integration_table = pd.DataFrame(tables_dict)
    mod_integration_table = mod_integration_table[mod_integration_table['Modality Integration'] != 'None']
    mod_integration_table = mod_integration_table.set_index(['Modality Integration', 'Modality', 'Algorithm'])
    mod_order = ['all'] + structural + functional
    mod_integration_table = mod_integration_table.reindex(mod_order, level=1)

    # save to excel
    os.makedirs('./tables', exist_ok=True)

    table.loc[structural].to_excel(f"tables/Table_{sample}_structural.xlsx")
    table.loc[functional].to_excel(f"tables/Table_{sample}_functional.xlsx")
    table.loc[environment_genetics].to_excel(f"tables/Table_{sample}_environment_genetics.xlsx")
    mod_integration_table.to_excel(f"tables/Table_{sample}_multi_modal.xlsx")

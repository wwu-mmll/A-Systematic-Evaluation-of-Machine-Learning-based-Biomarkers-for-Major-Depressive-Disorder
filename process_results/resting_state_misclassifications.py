import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scientific_plots as sp

from scipy.stats import spearmanr
from statsmodels.formula.api import ols, rlm
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from photonai.processing import ResultsHandler
from photonai.base import Hyperpipe

from process_results.scripts.misclassification_frequency import bootstrap_misclassification
from process_results.scripts.anova import AnovaES

plt.style.use(sp.get_style('mmll'))


# ----------------------------
# Set options
# ----------------------------

compute_misclassification = False
plot_corr = False
plot_violin = True
create_table = True
misclass_name = 'MF'

# ----------------------------
# Load and clean data
# ----------------------------

pheno = pd.read_csv('results_hc_mdd/resting_state/filter_hc_mdd/data_information/multi_modality.csv', na_values=-99)

# fix variables
pheno['Rem2'] = pheno['Rem'].copy()
pheno.loc[pheno['Rem2'] == 1, 'Rem2'] = 0

variables_to_rename = {'Alter': 'Age',
                       'BDI_Sum': 'BDI', 'HAMD_Sum21': 'HAMD',
                       'Hosp': 'Number_of_Hospitalizations'}
vars_hc = ['Age', 'BDI', 'HAMD', 'GAFscore']
vars_mdd = ['Age', 'BDI', 'HAMD', 'GAFscore',
            'Number_of_Hospitalizations']
cat_vars_hc = []
cat_vars_mdd = ['Rem', 'Rem2', 'Komorbid']

pheno = pheno.rename(columns=variables_to_rename)

# ----------------------------
# Compute misclassifactions
# ----------------------------
if compute_misclassification:

    handler = ResultsHandler()
    handler.load_from_file("results_hc_mdd/resting_state/filter_hc_mdd/pipeline_results/SVMPipeline/"
                           "svm_pipeline_results_2022-07-11_11-52-52/photon_result_file.json")
    best_config = handler.results.best_config

    # load data
    X = np.load('results_hc_mdd/resting_state/filter_hc_mdd/merger_data/X.npy')
    y = np.load('results_hc_mdd/resting_state/filter_hc_mdd/merger_data/y.npy')

    # load pipeline with best config
    pipeline = Hyperpipe.load_optimum_pipe("results_hc_mdd/resting_state/filter_hc_mdd/pipeline_results/SVMPipeline/"
                                       "svm_pipeline_results_2022-07-11_11-52-52/photon_best_model.photon")

    # use same number of splits as in all other analyses
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)

    # 100 bootstrap runs just like in the Nature paper
    n_boot = 100
    original_preds, bootstrap_preds = bootstrap_misclassification(X=X, y=y,
                                                                  photonai_pipeline=pipeline,
                                                                  n_boot=n_boot,
                                                                  cv=cv)

    # calculate balanced accuracy values for all bootstrap predictions
    bootstrap_baccs = list()
    for i in range(n_boot):
        bootstrap_baccs.append(balanced_accuracy_score(y_true=y, y_pred=bootstrap_preds[:, i]))

    # calculate balanced accuracy for the original predictions
    bacc = balanced_accuracy_score(y_true=y, y_pred=original_preds)

    # convert 2 to -1 so that it's easy to check whether the predictions are correct (by multiplying y_true and y_pred)
    y_new = y.copy()

    # all MDD patients and predictions will be set to -1
    y_new[y_new == 2] = -1
    bootstrap_preds[bootstrap_preds == 2] = -1

    # get misclassifications by multiplying the original labels with bootstrapped predictions
    # -1 * -1 = 1 (correct prediction), 1 * 1 = 1 (correct prediction)
    # -1 will always be an incorrect prediction
    misclassifications = np.tile(y_new[:, np.newaxis], n_boot) * bootstrap_preds

    # calculate misclassification frequency as the number of -1 values across bootstrap runs
    misclass_frequency = np.sum(misclassifications == -1, axis=1)

    pd.DataFrame({'y_true': y,
                  'y_pred': original_preds,
                  misclass_name: misclass_frequency}).to_csv('aggregated/misclass_frequency.csv')


if plot_violin:
    misclass = pd.read_csv('aggregated/misclass_frequency.csv')
    df = pd.concat([misclass, pheno], axis=1)

    sns.violinplot(data=df[df['Group'] == 2],
                   x='Rem',
                   y=misclass_name, cut=0, bw=.05)

    debug = True

# ----------------------------
# Create results table
# ----------------------------
if create_table:
    misclass = pd.read_csv('aggregated/misclass_frequency.csv')
    df = pd.concat([misclass, pheno], axis=1)

    def statistical_model(model_type: str, df_group: pd.DataFrame, variables: list, covariates: list = None):
        res = {'variable': [],
               'N': [],
               'r/beta': [],
               'F/t': [],
               'p': [],
               'covariates': []}

        for voi in variables:
            res['variable'].append(voi)

            if covariates:
                current_covariates = covariates.copy()
                for cov in current_covariates:
                    if voi in cov:
                        current_covariates.remove(cov)
            else:
                current_covariates = list()

            res['covariates'].append(current_covariates)

            if model_type == 'correlation':

                res['N'].append(np.sum(~np.isnan(df_group[voi])))
                r, p = spearmanr(df_group[misclass_name], df_group[voi], nan_policy='omit')
                res['r/beta'].append(np.round(r, 3))
                res['F/t'].append(None)
                res['p'].append(np.round(p, 3))

            elif model_type == 'anova':
                res['N'].append(np.sum(~np.isnan(df_group[voi])))
                aov = AnovaES(data=df_group, group_contrast=f'C({voi}, Sum)', covariates=current_covariates)
                aov_res = aov.anova_es(x='MF')
                res['p'].append(aov_res.loc[f'C({voi}, Sum)', 'p-unc'])
                f = aov_res.loc[f'C({voi}, Sum)', 'F']
                df1 = aov_res.loc[f'C({voi}, Sum)', 'DF']
                df2 = aov_res.loc['Residual', 'DF']
                res['F/t'].append(f"{f} [{df1}, {df2}]")
                res['r/beta'].append(None)

            elif model_type == 'lm':
                formula = f"MF ~ {voi}"
                for cov in covariates:
                    formula += f" + {cov}"

                lm = ols(formula, data=df_group).fit()
                #lm = rlm(formula, data=df_group).fit()

                res['N'].append(lm.resid.shape[0])
                res['r/beta'].append(lm.params[voi])
                res['F/t'].append(lm.tvalues[voi])
                res['p'].append(lm.pvalues[voi])
            else:
                raise NotImplementedError("Model Type not implemented. Use 'correlation', 'anova', or 'lm'.")

        return pd.DataFrame(res).set_index('variable')


    covariates = ['Age', 'C(Sex, Sum)', 'C(Dummy_BC_MR_pre)', 'C(Dummy_BC_MR_post)']
    df_mdd = df[df['Group'] == 2]
    df_hc = df[df['Group'] == 1]

    # MDD categorical variables
    res_mdd_anova = statistical_model(model_type='anova', df_group=df_mdd, variables=cat_vars_mdd)
    res_mdd_anova_covariates = statistical_model(model_type='anova', df_group=df_mdd,
                                                 variables=cat_vars_mdd, covariates=covariates)
    # MDD continuous variables
    res_mdd_correl = statistical_model(model_type='correlation', df_group=df_mdd, variables=vars_mdd)
    res_mdd_lm = statistical_model(model_type='lm', df_group=df_mdd, variables=vars_mdd,
                                   covariates=covariates)

    # HC categorical variables
    res_hc_anova = statistical_model(model_type='anova', df_group=df_hc, variables=cat_vars_hc)
    res_hc_anova_covariates = statistical_model(model_type='anova', df_group=df_hc,
                                                 variables=cat_vars_hc, covariates=covariates)
    # HC continuous variables
    res_hc_correl = statistical_model(model_type='correlation', df_group=df_hc, variables=vars_hc)
    res_hc_lm = statistical_model(model_type='lm', df_group=df_hc, variables=vars_hc,
                                   covariates=covariates)

    for res_mdd, res_hc, name in [(res_mdd_anova, res_hc_anova, "_anova"),
                                  (res_mdd_anova_covariates, res_hc_anova_covariates, "_anova_covariates"),
                                  (res_mdd_correl, res_hc_correl, "_spearman"),
                                  (res_mdd_lm, res_hc_lm, "_lm_covariates")]:
        res = pd.concat({'MDD': res_mdd, 'HC': res_hc}, axis=1)
        res.to_excel(f'tables/misclassification_frequency{name}.xlsx', engine='xlsxwriter')

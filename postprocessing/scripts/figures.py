import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.stats import spearmanr

import scientific_plots as sp
from scientific_plots.plots import error_bars

plt.style.use(sp.get_style('mmll'))


def plot_ensemble_hist(file, sample, filename):
    df = pd.read_csv(file)
    df = df[df['Sample'] == sample]
    sns.histplot(data=df, x='proba', hue='Group', multiple='dodge',
                 kde=True, palette='colorblind')
    sp.savefig(filename)
    plt.show()


def plot_correlations(df, sample, variables_of_interest, y, filename, plot_type='scatter', hue=None):
    if sample:
        df = df[df['Sample'] == sample]

    n_vars = len(variables_of_interest)
    row = np.ceil(n_vars / 3).astype(int)
    if n_vars <= 3:
        col = n_vars
    else:
        col = 3
    cm = 1 / 2.54
    fig, axes = plt.subplots(row, col, figsize=(18.3 * cm, 10 * cm), dpi=300)
    axes = axes.reshape(-1)

    for i, voi in enumerate(variables_of_interest):
        if plot_type == 'scatter':
            sns.scatterplot(data=df, x=voi, hue=hue, y=y, ax=axes[i],
                            palette='colorblind')
        elif plot_type == 'regression':
            print(voi)
            r = df[[voi, y]].corr(method='spearman').iloc[0, 1]
            p = spearmanr(df[y], df[voi], nan_policy='omit')[1]
            if p < 0.001:
                p_str = "$\it{{p}}$ < 0.001".format(p)
            elif p < 0.01:
                p_str = "$\it{{p}}$ = {:.3f}".format(p)
            else:
                p_str = "$\it{{p}}$ = {:.2f}".format(p)
            reg = sns.regplot(data=df, x=voi, y=y, ax=axes[i], marker='.', color='black',
                              scatter_kws={'color': 'black', 'facecolor': 'black', 'alpha': 0.3},
                              line_kws={'color': 'red'})
            reg.set_ylim(0, 100)
        else:
            raise NotImplementedError
    sns.despine()
    plt.tight_layout()
    sp.savefig(filename)
    plt.show()


def plot_roc(sample: str, filename: str):
    df = pd.read_csv('aggregated/predictions_ensemble.csv')
    df = df[df['Sample'] == sample]

    y_test_1 = df['Group'] == 1
    y_test_2 = df['Group'] == 2
    y_test = np.stack([y_test_1, y_test_2])
    y_test = y_test.transpose().astype(int)

    y_score_1 = 1 - df['proba'].to_numpy()
    y_score_2 = df['proba'].to_numpy()
    y_score = np.stack([y_score_1, y_score_2])
    y_score = y_score.transpose()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 2
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(
        fpr['micro'],
        tpr['micro'],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc['micro'],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    sp.savefig(filename)
    plt.show()


def point_plot(df, modalities: dict, filename: str = 'pointplot.pdf', ncols: int = 4, nrows: int = 4):
    fig = plt.figure(figsize=(14, 10), constrained_layout=False, dpi=300)
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows)

    row = 0
    col = 0
    for mod, name in modalities.items():
        mod_df = df[df['Modality'] == mod]
        ax = fig.add_subplot(spec[row, col])
        with plt.rc_context({'lines.linewidth': 1}):
            g = sns.pointplot(data=mod_df, x="PipelineType", y="Folds", ci="sd",
                              hue="Sample", dodge=0.3,
                              legend=False, height=5, aspect=2, ax=ax, lw=5,
                              capsize=0.1, palette="Set2", order=["BoostingPipeline", "KNNPipeline",
                                                                  "LogisticRegressionPipeline", "NaiveBayesPipeline",
                                                                  "RandomForestPipeline", "SVMPipeline"])

        ax.set(xlabel=None, ylabel='Balanced Accuracy', xticklabels=["Boosting", "KNN", "LR", "NB", "RF", "SVM"])

        plt.xticks(rotation=35, horizontalalignment='right')
        plt.title(name, fontsize=16)
        plt.setp(ax.collections, sizes=[12])
        g.set(ylim=(0.4, 1))
        plt.axhline(0.5, color='k')
        plt.axhline(0.6, color=(110 / 255, 110 / 255, 110 / 255, 0.4), linestyle='dotted')
        plt.axhline(0.7, color=(110 / 255, 110 / 255, 110 / 255, 0.4), linestyle='dotted')
        plt.axhline(0.8, color=(110 / 255, 110 / 255, 110 / 255, 0.4), linestyle='dotted')
        plt.axhline(0.9, color=(110 / 255, 110 / 255, 110 / 255, 0.4), linestyle='dotted')

        handles, labels = ax.get_legend_handles_labels()

        # reverse the order
        ax.legend(handles[::-1], labels[::-1])

        ax.get_legend().remove()
        col += 1
        if col == ncols:
            col = 0
            row += 1
    plt.tight_layout()
    ax.legend(handles, ["HC vs MDD", "HC vs aMDD", "HC vs rMDD"], loc=(1.1, 0.3))

    sp.savefig(filename)
    plt.show()


def point_plot_vertical(df, sections, plot_name: str, x_variable: str = 'BACC',
                        x_label: str = 'Balanced Accuracy'):
    df_mean_idx = df.groupby(by=['AnalysisName', 'PipelineType', 'Modality']).mean().reset_index().groupby(
        by='AnalysisName').idxmax()['BACC']
    df_mean = df.groupby(by=['AnalysisName', 'PipelineType', 'Modality']).mean().reset_index()
    best_analyses = df_mean.iloc[df_mean_idx]

    d = pd.DataFrame()
    for key, row in best_analyses.iterrows():
        d = pd.concat([d, df[(df['AnalysisName'] == row['AnalysisName']) & (df['Modality'] == row['Modality']) & (
                df['PipelineType'] == row['PipelineType'])]])

    fig, ax, cax, = error_bars(x=x_variable, y='AnalysisName', sections=sections,
                               data=d, central_tendency='mean',
                               error_method='standard_deviation', xlim=(0.45, 1),
                               cax_percentage="77%")
    fig.set_size_inches(9 / 2.54, 11 / 2.54)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
    y_min = ax.get_ylim()[0]
    ax.vlines([0.5, 0.5], y_min + 0.02, 0.23, color='#DCDCDC', linestyle='-', lw=1)
    ax.vlines([0.6, 0.6], y_min + 0.02, 0.23, color='#DCDCDC', linestyle='-', lw=1)
    ax.vlines([0.7, 0.7], y_min + 0.02, 0.23, color='#DCDCDC', linestyle='-', lw=1)
    ax.vlines([0.8, 0.8], y_min + 0.02, 0.23, color='#DCDCDC', linestyle='-', lw=1)
    ax.vlines([0.9, 0.9], y_min + 0.02, 0.23, color='#DCDCDC', linestyle='-', lw=1)
    ax.set_xlabel(x_label)

    sp.savefig(plot_name)
    plt.show()
    return


def plot_reliability_correction(df, sample, filename, modalities, colors=None):
    df_sample = df[df['Sample'] == sample]

    # keep only pipelines with largest BACC
    df_sample = df_sample.loc[df_sample.groupby(by=['AnalysisName']).idxmax()['BACC']]

    df_sample = df_sample[df_sample['AnalysisName'].isin(modalities)]

    # wide to long format
    df_long = pd.wide_to_long(df_sample, ['BACC_corrected_simulated_', 'MCC_corrected_simulated_'],
                              i=['AnalysisName', 'PipelineType'], j='Reliability',
                              suffix='.*', sep='').reset_index()
    df_long_sd = df_long[df_long['Reliability'].str.slice(4) == "sd"]
    df_long = df_long[df_long['Reliability'].str.slice(4) != "sd"]
    df_long['Reliability'] = df_long['Reliability'].astype(float)

    fig, ax = plt.subplots(figsize=(9 / 2.54, 5 / 2.54))
    if colors is None:
        colors = sns.color_palette(palette='deep', n_colors=len(modalities))

    # get order first
    sorted_mods = df_long[df_long['Reliability'] == 1].sort_values(by=['BACC_corrected_simulated_'])['AnalysisName']
    for i, mod in enumerate(sorted_mods):
        x = df_long.loc[df_long['AnalysisName'] == mod, 'Reliability'].to_numpy()
        y = df_long.loc[df_long['AnalysisName'] == mod, 'BACC_corrected_simulated_'].to_numpy()

        ax.plot(x, y, color=colors[i])
        arrowprops = dict(arrowstyle='-', color=colors[i], linewidth=0.5, relpos=(0, 0), shrinkA=0, shrinkB=3)
        ax.annotate(text=mod, xy=(0.1, np.max(y)), xycoords='data',
                    xytext=(0.05, (i / 16) + 0.5), textcoords='data',
                    arrowprops=arrowprops,
                    horizontalalignment='left', verticalalignment='center'
                    )

    ax.set(ylim=(0.5, 1))
    ax.set(xlim=(0.1, 1))
    ax.set(xticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.invert_xaxis()
    ax.set(ylabel='Attenuation Corrected BACC')
    ax.set(xlabel='Simulated Reliability')
    plt.tight_layout()
    sp.savefig(filename)
    plt.show()
    return


def plot_reliability_correction_grid(df, sample, filename):
    df_sample = df[df['Sample'] == sample]

    # keep only pipelines with largest BACC
    df_sample = df_sample.loc[df_sample.groupby(by=['AnalysisName']).idxmax()['BACC']]
    mods = ['Freesurfer', 'VBM',
            'RS Connectivity', 'ALFF', 'fALFF', 'LCOR',
            'RS Network Parameters',
            'Hariri',
            'FA', 'MD', 'DTI Network Parameters']
    df_sample = df_sample[df_sample['AnalysisName'].isin(mods)]

    # wide to long format
    df_long = pd.wide_to_long(df_sample, ['BACC_corrected_simulated_', 'MCC_corrected_simulated_'],
                              i=['AnalysisName', 'PipelineType'], j='Reliability',
                              suffix='.*', sep='').reset_index()
    df_long_sd = df_long[df_long['Reliability'].str.slice(4) == "sd"]
    df_long = df_long[df_long['Reliability'].str.slice(4) != "sd"]
    df_long['Reliability'] = df_long['Reliability'].astype(float)

    ncols = 4
    nrows = 3
    fig = plt.figure(figsize=(18 / 2.54, 13 / 2.54), constrained_layout=False, dpi=300)
    spec = fig.add_gridspec(ncols=ncols, nrows=nrows)
    row = 0
    col = 0
    colors = sns.color_palette(palette='deep', n_colors=len(mods))

    # get order first
    sorted_mods = df_long[df_long['Reliability'] == 1].sort_values(by=['BACC_corrected_simulated_'])['AnalysisName']
    for i, mod in enumerate(sorted_mods):
        x = df_long.loc[df_long['AnalysisName'] == mod, 'Reliability'].to_numpy()
        y = df_long.loc[df_long['AnalysisName'] == mod, 'BACC_corrected_simulated_'].to_numpy()
        lower = y - df_long_sd.loc[df_long_sd['AnalysisName'] == mod, 'BACC_corrected_simulated_'].to_numpy()
        upper = y + df_long_sd.loc[df_long_sd['AnalysisName'] == mod, 'BACC_corrected_simulated_'].to_numpy()

        ax = fig.add_subplot(spec[row, col])
        ax.plot(x, y, color=colors[i])
        ax.plot(x, lower, color=colors[i], alpha=0.1)
        ax.plot(x, upper, color=colors[i], alpha=0.1)
        ax.fill_between(x, lower, upper, alpha=0.2)

        ax.set(title=mod)
        ax.set(ylim=(0.5, 1))
        ax.set(xlim=(0.1, 1))
        ax.set(xticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        ax.invert_xaxis()
        ax.set(ylabel='Corrected Balanced Accuracy')
        ax.set(xlabel='Simulated Reliability')
        col += 1
        if col == ncols:
            col = 0
            row += 1
    plt.tight_layout()
    sp.savefig(filename)
    plt.show()
    return
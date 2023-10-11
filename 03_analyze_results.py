import pandas as pd
from os.path import join

from svgutils.compose import Figure, SVG
import cairosvg

from postprocessing.scripts.utils import aggregate_results, create_ensemble_preds, create_results_table, \
    reliability_corrected_correlations, create_descriptives_statistics_table
from postprocessing.scripts.figures import point_plot_vertical, plot_reliability_correction, \
    plot_reliability_correction_grid


# ---------------------
# 1 Results Setup
# ---------------------

# define results and data paths
results_folder = './results'
data_folder = './results'

# specify the subsamples that should be used
filter_names = ["filter_hc_mdd"]

# specify photonai pipelines
pipelines = ["BoostingPipeline", "KNNPipeline", "LogisticRegressionPipeline", "NaiveBayesPipeline",
             "RandomForestPipeline", "SVMPipeline"]

aggregate = True
run_ensemble = True
run_reliability_correction = True

create_tables = True
create_samples_statistics = True

plot_corr = True
plot_figure_2 = True
create_reliability_plots = True

# --------------------------------
# 2 Aggregate and process results
# --------------------------------
# 2.1 this step creates three different csv files containing the results
#       a) results.csv - classification metrics for all folds
#       b) results_aggregated.csv - mean and std of classification metrics
#       c) predictions.csv - long-format table containing all test predictions
if aggregate:
    aggregate_results(results_folder=results_folder, filter_names=filter_names, pipeline_types=pipelines,
                      data_folder=data_folder)

# 2.2 create ensemble predictions
if run_ensemble:
    create_ensemble_preds(file='aggregated/predictions.csv', add_to_results=True,
                          samples=filter_names)


# 2.3 correct classification metrics for imperfect reliability
if run_reliability_correction:
    df1 = pd.read_csv('aggregated/predictions.csv')
    df2 = pd.read_csv('aggregated/predictions_ensemble.csv')
    df = pd.concat([df1, df2], axis=0)

    reliability_corrected_correlations(predictions=df,
                                       save_to_file='aggregated/results_reliability.csv')

# ---------------------------------------
# 3 Create results tables for manuscript
# ---------------------------------------
if create_samples_statistics:
    create_descriptives_statistics_table(save_to_file="tables/descriptive_statistics.xlsx")
    create_descriptives_statistics_table(save_to_file="tables/descriptive_statistics_acute.xlsx",
                                         sample_filter='filter_hc_mdd_acute')
    create_descriptives_statistics_table(save_to_file="tables/descriptive_statistics_recurrent.xlsx",
                                         sample_filter='filter_hc_mdd_severe')
    create_descriptives_statistics_table(save_to_file="tables/descriptive_statistics_age_24_28.xlsx",
                                         sample_filter='filter_hc_mdd_age_24_28')

if create_tables:
    for sample in filter_names:
        create_results_table(file='aggregated/results_mean_sd_update.csv', sample=sample)

# ---------------------------------------
# 4 Create figures for manuscript
# ---------------------------------------
# Figure 2
if plot_figure_2:
    sections = {'T1-weighted MRI': ['Freesurfer', 'VBM'],
                'Resting-State fMRI': ['RS Connectivity', 'ALFF', 'fALFF', 'LCOR',
                                       'RS Network Parameters'],
                'Task-based fMRI': ['Hariri'],
                'DTI': ['FA', 'MD', 'DTI Network Parameters'],
                'Multimodal Integration': ["Data Integration", "Ensemble", "Modality Ensemble",
                                           "Algorithm Ensemble"],
                'Genetics': ['PRS'],
                'Environment': ["Social Support", "Childhood Maltreatment"],
                }
    df = pd.read_csv('aggregated/results_update.csv')
    for sample in filter_names:
        point_plot_vertical(df[df['Sample'] == sample],
                            sections=sections,
                            plot_name=f"plots/predictive_performance/accuracy_{sample}.png")

if create_reliability_plots:
    sections = {'T1-weighted MRI': ['Freesurfer', 'VBM'],
                'Resting-State fMRI': ['RS Connectivity', 'ALFF', 'fALFF', 'LCOR',
                                       'RS Network Parameters'],
                'Task-based fMRI': ['Hariri'],
                'DTI': ['FA', 'MD', 'DTI Network Parameters'],
                'Multimodal Integration': ["Data Integration", "Ensemble", "Modality Ensemble",
                                           "Algorithm Ensemble"],
                'Genetics': ['PRS'],
                }
    df = pd.read_csv('aggregated/results_reliability.csv')
    point_plot_vertical(df[df['Sample'] == 'filter_hc_mdd'],
                        sections=sections,
                        plot_name=f"plots/reliability/empirical_corrected_accuracy_hc_mdd.png",
                        x_variable='BACC_corrected_empirical_pheno',
                        x_label='Attenuation Corrected Balanced Accuracy')

    # ===============
    # Figure 3b
    # ===============
    import seaborn as sns
    c = sns.color_palette('colorblind', 5)

    df = pd.read_csv('aggregated/results_reliability_mean_sd.csv')

    # ===================
    # Print some results
    # ===================
    modalities = ['Freesurfer', 'VBM',
            'RS Connectivity', 'ALFF', 'fALFF', 'LCOR',
            'RS Network Parameters',  'Hariri',
            'FA', 'MD', 'DTI Network Parameters']
    df_current = df[(df['AnalysisName'].isin(modalities)) & (df['Sample'] == 'filter_hc_mdd')]

    rs_bacc_emp = df_current[df_current['Modality'] == 'RS Connectivity'][['BACC_corrected_empirical_pheno',
                                                                           'BACC_corrected_empirical_pheno_sd',
                                                                           'PipelineType']]
    rs_bacc_emp = rs_bacc_emp.sort_values(by='BACC_corrected_empirical_pheno')

    ens_bacc_emp = df[(df['AnalysisName'] == 'Ensemble')& (df['Sample'] == 'filter_hc_mdd')][['BACC_corrected_empirical_pheno',
                                                                           'BACC_corrected_empirical_pheno_sd',
                                                                           'PipelineType']]
    ens_bacc_emp = ens_bacc_emp.sort_values(by='BACC_corrected_empirical_pheno')

    rs_bacc_0_1 = df_current[df_current['Modality'] == 'RS Connectivity']['BACC_corrected_simulated_0.1']
    rs_bacc_0_1 = rs_bacc_0_1.sort_values().iloc[-1]
    rs_bacc_0_3 = df_current[df_current['Modality'] == 'RS Connectivity']['BACC_corrected_simulated_0.3']
    rs_bacc_0_3 = rs_bacc_0_3.sort_values().iloc[-1]

    all_bacc_0_1 = df_current[['BACC_corrected_simulated_0.1', 'Modality']].groupby(by='Modality').max()

    print("RS Conn:")
    print(f"Corrected BACC (Rel=0.1) = {rs_bacc_0_1}")
    print(f"Corrected BACC (Rel=0.3) = {rs_bacc_0_3}")
    print(f"Corrected BACC (Rel=empirical) = {rs_bacc_emp.iloc[-1,:]}")
    print("Ensemble")
    print(f"Corrected BACC (Rel=empirical) = {ens_bacc_emp.iloc[-1,:]}")
    print()
    print(f"All modalities (Rel=0.1) = {all_bacc_0_1}")

    mods = ['Freesurfer', 'VBM',
            'RS Connectivity', 'ALFF', 'fALFF', 'LCOR',
            'RS Network Parameters']
    plot_reliability_correction(df, sample='filter_hc_mdd',
                                filename='plots/reliability/simulated_corrected_accuracy_upper_panel_hc_mdd.png',
                                modalities=mods,
                                colors=[c[0], c[0], c[1], c[1], c[1], c[1], c[1]])

    mods = [ 'Hariri',
            'FA', 'MD', 'DTI Network Parameters',
             "Data Integration", "Ensemble", "Modality Ensemble",
             "Algorithm Ensemble"
             ]
    plot_reliability_correction(df, sample='filter_hc_mdd',
                                filename='plots/reliability/simulated_corrected_accuracy_lower_panel_hc_mdd.png',
                                modalities=mods,
                                colors=[c[3], c[3], c[3], c[2], c[4], c[4], c[4], c[4]])

    from svgutils.compose import Figure, SVG, Panel, Text

    svg = SVG("plots/reliability/empirical_corrected_accuracy_hc_mdd/empirical_corrected_accuracy_hc_mdd.svg",
              fix_mpl=True)

    Figure(2.3 * svg.width, svg.height,
           Panel(
               SVG("plots/reliability/empirical_corrected_accuracy_hc_mdd/empirical_corrected_accuracy_hc_mdd.svg",
                   fix_mpl=True),
               Text("A", -10, 20, size=12, weight='bold')
           ).move(20, 0),
           Panel(
               SVG("plots/reliability/simulated_corrected_accuracy_upper_panel_hc_mdd/"
                   "simulated_corrected_accuracy_upper_panel_hc_mdd.svg", fix_mpl=True),
               Text("B", -10, 20, size=12, weight='bold')
           ).move(290, 30),
           Panel(
               SVG("plots/reliability/simulated_corrected_accuracy_lower_panel_hc_mdd/"
                   "simulated_corrected_accuracy_lower_panel_hc_mdd.svg", fix_mpl=True)
           ).move(290, 150)
           ).save("plots/reliability/reliability_corrected_composite_plot.svg")

    import cairosvg

    cairosvg.svg2pdf(url="plots/reliability/reliability_corrected_composite_plot.svg",
                     write_to='plots/reliability/reliability_corrected_composite_plot.pdf',
                     background_color='white')
    cairosvg.svg2png(url="plots/reliability/reliability_corrected_composite_plot.svg",
                     write_to="plots/reliability/reliability_corrected_composite_plot.png",
                     output_width=1400, background_color='white', dpi=300)

    plot_reliability_correction_grid(df, sample='filter_hc_mdd',
                                     filename='plots/reliability/simulated_corrected_accuracy_grid_hc_mdd.png')


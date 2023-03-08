import pandas as pd


pipes = pd.read_csv('pipeline_types_hc_mdd.csv', header=None)
samples = pd.read_csv('sample_filter_hc_mdd.csv', header=None)

unique_combinations = []

for sample in samples[0].tolist():
    for pipeline in pipes[0].tolist():
        unique_combinations.append([sample, pipeline])

pd.DataFrame(unique_combinations).to_csv('jobs.csv', header=False, index=False)

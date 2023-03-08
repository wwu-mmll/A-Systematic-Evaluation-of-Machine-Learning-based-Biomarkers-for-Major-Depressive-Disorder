from macs_datahub.main import run_pipeline


config_list = ['./analyses/prs/prs.yaml']

for config in config_list:
    run_pipeline(config_file_dir=config)
debug = True

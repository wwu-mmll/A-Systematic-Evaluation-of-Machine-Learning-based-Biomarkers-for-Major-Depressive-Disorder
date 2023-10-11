import sys
sys.path.append('./macs_datahub')
from macs_datahub.main import run_pipeline


run_pipeline(config_file_dir='./analyses/dummy_modality.yaml')

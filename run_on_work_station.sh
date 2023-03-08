#!/bin/bash

# use this script to start all analyses at once on a local Linux machine
python macs_datahub/main.py  --yaml_file analyses/configs/prs.yaml
python macs_datahub/main.py  --yaml_file analyses/configs/social_support.yaml
python macs_datahub/main.py  --yaml_file analyses/configs/childhood_maltreatment.yaml
python macs_datahub/main.py  --yaml_file analyses/configs/freesurfer.yaml
python macs_datahub/main.py  --yaml_file analyses/configs/dti_fa.yaml
python macs_datahub/main.py  --yaml_file analyses/configs/dti_md.yaml
python macs_datahub/main.py  --yaml_file analyses/configs/graph_metrics_rs.yaml
python macs_datahub/main.py  --yaml_file analyses/configs/graph_metrics_dti.yaml
python macs_datahub/main.py  --yaml_file analyses/configs/rs.yaml
python macs_datahub/main.py  --yaml_file analyses/configs/alff.yaml
python macs_datahub/main.py  --yaml_file analyses/configs/falff.yaml
python macs_datahub/main.py  --yaml_file analyses/configs/lcor.yaml
python macs_datahub/main.py  --yaml_file analyses/configs/cat12.yaml
python macs_datahub/main.py  --yaml_file analyses/configs/hariri.yaml
python macs_datahub/main.py  --yaml_file analyses/configs/multi_modality.yaml



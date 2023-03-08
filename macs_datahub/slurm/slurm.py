import os
from datetime import datetime
import pandas as pd
from itertools import product


class SlurmJobCreator:
    def __init__(self, filter_file: str, pipeline_file: str, conda_env: str,
                 logging_folder: str, partition: str,
                 n_dask_processes: int, mail: str, type_calc: str):
        self.filter_file = filter_file
        self.pipeline_file = pipeline_file
        self.conda_env = conda_env
        self.logging_folder = logging_folder
        self.partition = partition
        self.n_dask_processes = n_dask_processes
        self.mail = mail
        self.type_calc = type_calc

        self.cmd = None
        self.fid = None

    def create(self,
               job_name: str,
               yaml_file: str,
               filename: str = 'job.cmd',
               time: str = '0-00:15:00',
               memory: int = 2,
               debug: bool = False):
        """
        Create Palma SLURM cmd script to run python job
        :param job_name: name of your job
        :param yaml_file: name of yaml configuration file
        :param filter_file: .csv file that specifies all sample filters that should be used
        :param conda_env: existing conda environment on Palma
        :param filename: specify the name of the .cmd file
        :param logging_folder: specify folder to write job output to
        :param partition: specify Palma partition (normal, express, ...)
        :param time: has to follow SLURM convention, i.e. 0-00:00:00 (day-hours:minutes:seconds)
        :param n_dask_processes: number of cpu cores to be used in PHOTONAI parallelization
        :param memory: GB working memory per task
        :param mail: specify your ziv email address if you want to be notified
        :param debug: if True, only prints created script to console, otherwise file is written
        :return:
        """
        self._create_combined_job_csv()
        self._get_number_of_jobs(filter_file=self.job_file)
        self._initialize(filename=filename)
        self._add_job_infos(job_name=job_name, output_folder=self.logging_folder, mail=self.mail)
        self._add_job_specs(partition=self.partition,
                            memory=memory, time=time,
                            n_dask_processes=self.n_dask_processes)
        self._add_env(conda_env=self.conda_env)
        self._add_script(conda_env=self.conda_env, yaml_file=yaml_file)
        self._write(debug=debug)

    def _create_combined_job_csv(self):
        filters = pd.read_csv(self.filter_file, header=None)[0].tolist()
        pipelines = pd.read_csv(self.pipeline_file, header=None)[0].tolist()
        combinations = list(product(filters, pipelines))
        df = pd.DataFrame(combinations, columns=["filter", "pipeline"])
        self.job_file = os.path.join(os.path.dirname(self.filter_file), "jobs_{}.csv".format(self.type_calc))
        df.to_csv(self.job_file, index=False, header=False)

    def _get_number_of_jobs(self, filter_file: str):
        df = pd.read_csv(filter_file, header=None)
        self.n_jobs = df.shape[0]

    def _write(self, debug):
        if debug:
            print(self.cmd)
        else:
            self.fid.write(self.cmd)
            self.fid.close()

    def _initialize(self, filename):
        self.fid = open(filename, 'w')
        self.cmd = "#!/bin/bash\n"
        self.cmd += """
# =======================
# BATCH JOB PALMA
# created: {}
# =======================
""".format(datetime.now())

    def _add_job_infos(self, job_name, output_folder, mail):
        self.cmd += """
#SBATCH --job-name={}
#SBATCH --output={}""".format(job_name, os.path.join(output_folder, job_name + "_job_%a.dat"))
        if mail:
            self.cmd += "\n#SBATCH --mail-type=ALL"
            self.cmd += "\n#SBATCH --mail-user={}".format(mail)

    def _add_job_specs(self, partition, n_dask_processes, memory, time):
        self.cmd += """

#SBATCH --partition {}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={}
#SBATCH --mem-per-cpu={}G
#SBATCH --time={}
#SBATCH --array=1-{}
            """.format(partition, n_dask_processes, memory, time, self.n_jobs)

    def _add_env(self, conda_env):
        self.cmd += """
# add python
module load Miniconda3

# activate conda env
eval "$(conda shell.bash hook)"
conda activate {}        
            """.format(conda_env)

    def _add_script(self, conda_env, yaml_file):
        self.cmd += """
pipeline=$(sed -n $SLURM_ARRAY_TASK_ID'p' {} | cut -d ',' -f2)
filter=$(sed -n $SLURM_ARRAY_TASK_ID'p' {} | cut -d ',' -f1)

# run script
{}/bin/python ../macs_datahub/main.py  --yaml_file {} --sample_filter $filter --pipeline $pipeline""".format(
            self.job_file, self.job_file, conda_env, yaml_file)



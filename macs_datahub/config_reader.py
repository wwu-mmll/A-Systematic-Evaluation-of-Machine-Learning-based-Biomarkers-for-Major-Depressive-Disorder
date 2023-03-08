import os
import yaml


class ConfigReader:

    """ Read a configuration from a yaml file """

    @staticmethod
    def execute(config_path: str = None):

        """ This function tries to load the configuration from a yaml file

        Parameters
        ----------
        config_path: str
            path to the configuration file (yaml)

        Returns
        -------
        configuration for the pipeline
        """

        if config_path is None:
            raise ValueError("No config file provided")
        if not os.path.isfile(config_path):
            raise ValueError("Could not find config file in path provided")

        stream = open(config_path, 'r')
        return yaml.load(stream, Loader=yaml.SafeLoader)

from email.policy import default
import os 
import argparse
import logging
import yaml


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def main(config_path, datasource):
    config = read_params(config_path)
    print(config)




if __name__ == '__main__':

    # creating a parser object 
    arg = argparse.ArgumentParser()  

    # getting the default path of the yaml file containing all the parameters
    default_config_path = os.path.join("config","params.yaml")

    # adding a argument for the config folder path
    arg.add_argument("--config", "-c", default=default_config_path)

    # adding a argument for the datasource 
    arg.add_argument("--datasource", "-d", default=None)

    parsed_args = arg.parse_args()

    main(config_path=parsed_args.config, datasource=parsed_args.datasource)

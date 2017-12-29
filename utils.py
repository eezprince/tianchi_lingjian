import yaml


def parse_config(config_path):
    with open(config_path) as stream:
        return yaml.load(stream)

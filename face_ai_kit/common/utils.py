"""
Description: Utils

Author:
Date Created: Dec 26, 2023
Date Modified: Dec 26, 2023
License: MIT License
"""


import yaml


def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)
    return loaded
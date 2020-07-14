import json
import argparse


def parse_params(description, parameter_file='../projects/nuclear/nuclear.json'):
    parser = argparse.ArgumentParser(description)
    parser.add_argument('--params',
                        help='The location of the parameters file.',
                        default=parameter_file)

    with open(parser.parse_args().params, 'r') as f:
        params = json.load(f)

    return params

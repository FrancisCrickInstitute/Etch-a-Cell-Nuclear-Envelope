"""
010_PREPROCESS_CSV

The Zooniverse web system for producing machine learning annotations has a CSV output format. These
CSVs can be very large and include a good deal of data which is not relevant for training a model.
This file breaks the raw CSV down in to smaller CSVs on a per image basis, so that they easily fit
in memory. It also filters data unnecessary for training.
"""
import os
from src.param_parser import parse_params
from src.zooniverse import ZooniverseCSVParser


# problem if annotation has deviant resolution - a slice can only have one resolution
# in case (consensus) resolution is different from others in stack, numpy array operations won't be possible
# TODO: handle these types of errors


def preprocess_zooniverse_csv(output_dir, input_path, workflow):
    zoon_parser = ZooniverseCSVParser(output_dir=output_dir, workflow=workflow)
    zoon_parser.convert(zooniverse_csv_path=input_path)

    print('Finished converting Zooniverse csv...')
    print(f'Total processed rows in workflow (including failures): {zoon_parser.processed}')
    print(f'Missing reference image failures: {zoon_parser.errors["missing_ref_image"]}')
    print(f'Invalid (annotation) format failures: {zoon_parser.errors["invalid_format"]}')


if __name__ == '__main__':
    params = parse_params("Run step 010 to break the zooniverse csv apart in to smaller files.")

    csv_input_path = os.path.join('..', params['zooniverse_csv_file'])
    csv_output_dir = os.path.join('..', params['processed_csv_dir'])

    zooniverse_workflow = params['zooniverse_workflow']

    preprocess_zooniverse_csv(csv_output_dir, csv_input_path, zooniverse_workflow)


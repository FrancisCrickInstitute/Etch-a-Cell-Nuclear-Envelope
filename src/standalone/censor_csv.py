import os
from src.param_parser import parse_params
from src.zooniverse import ZooniverseCSVParser


params = parse_params("Create censored zooniverse csv file.", parameter_file='../../projects/nuclear/nuclear.json')

csv_input_path = os.path.join('../..', params['zooniverse_csv_file'])
csv_output_dir = os.path.join('../..', params['processed_csv_dir'], '..')
csv_output_filename = os.path.join(csv_output_dir, 'censored.csv')

zooniverse_workflow = params['zooniverse_workflow']

zoon_parser = ZooniverseCSVParser(output_dir=csv_output_dir, workflow=zooniverse_workflow)
zoon_parser.censor(zooniverse_csv_path=csv_input_path, output_path=csv_output_filename)

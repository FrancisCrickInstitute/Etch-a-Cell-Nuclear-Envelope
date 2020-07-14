"""
PREDICT
"""
import os
import shutil
import argparse
import json

from src.ml.DataLoader import DataLoader
from src.ml.model import load_latest_model
from src.ml.model_predict import model_predict


def predict(params, images_dir, predictions_dir, model_dir, aggregation_method, model_size_xy_nm, model_size_z_nm,
            model_name=None, clear_existing=False, tri_axis=False):
    if clear_existing and os.path.exists(predictions_dir):
        shutil.rmtree(predictions_dir)
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)

    if model_name is None:
        data_loader = DataLoader(params, images_dir, predictions_dir, model_dir, aggregation_method)
        model, epoch = load_latest_model(data_loader.get_name())
    else:
        model, epoch = load_latest_model(model_name)
    if epoch != 0:
        model_predict(model, images_dir, predictions_dir, model_size_xy_nm, model_size_z_nm, tri_axis=tri_axis)
    else:
        raise Exception("Model load error")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run to run model prediction.")
    parser.add_argument('--source',
                        help='The location of the source images.',
                        default='projects/nuclear/resources/images/raw-stacks')
    parser.add_argument('--predictions',
                        help='The location of the predictions images.',
                        default='projects/nuclear/resources/images/predictions-stacks')
    parser.add_argument('--tri_axis',
                        help='Run predictions across all 3 axes (x, y, z) and then recombine for the final prediction.',
                        default=False)

    args = parser.parse_args()
    tri_axis = args.tri_axis
    with open(args.params, 'r') as f:
        params = json.load(f)

    images_dir = os.path.join("..", parser.parse_args().source)
    predictions_dir = os.path.join("..", parser.parse_args().predictions)

    model_dir = os.path.join('..', params['model']['save_dir'])
    model_size_xy_nm = params['target_xy_nm']
    model_size_z_nm = params['target_z_nm']
    aggregation_method = params['aggregation_method']

    predict(params, images_dir, predictions_dir, model_dir, aggregation_method,
            model_size_xy_nm, model_size_z_nm, tri_axis=tri_axis)

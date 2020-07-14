"""
080_TRAIN

This file trains the machine learning model.
"""
from src.ml.DataLoader import DataLoader
from src.ml.model import get_model
from src.ml.model_train import model_train
from src.param_parser import parse_params


def train_model(params, model_images_dir, model_labels_dir, model_save_dir, aggregation_method):
    data_loader = DataLoader(params, model_images_dir, model_labels_dir, model_save_dir, aggregation_method)
    print("Training data:   ", data_loader.training_rois)
    print("Validation data: ", data_loader.validation_rois)
    print("Holdout data:    ", data_loader.holdout_rois)
    #data_loader.visualise_training_batch()

    model, epoch = get_model(params, data_loader)
    model_train(model, params, data_loader, epoch)


if __name__ == '__main__':
    params = parse_params("Run step 080 to train the model.")

    model_images_dir = '../'+params['data']['images_dir']
    model_labels_dir = '../'+params['data']['labels_dir']
    model_save_dir = '../'+params['model']['save_dir']
    aggregation_method = params['aggregation_method']

    train_model(params, model_images_dir, model_labels_dir, model_save_dir, aggregation_method)

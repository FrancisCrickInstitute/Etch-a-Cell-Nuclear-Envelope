"""
050_CROP.py

This file crops a cell image such that only the portions which contain annotated nucleus remain.
The citizen science volunteers were instructed only to annotate the central nucleus in an image,
but cells are packed close enough together such that some parts of neighbouring nucleus are also
visible. In order not to confuse the model with bits of unannotated nuclear membrane, we remove
as best as possible those parts from the training data.
"""
import os
import shutil

from tqdm import tqdm

from src.param_parser import parse_params
from src.cropping import get_rois, crop_roi


def do_crops(image_folder, label_folder, cropped_image_folder, cropped_label_folder,
             patch_size, padding, clear_existing=False):
    if clear_existing and os.path.exists(cropped_image_folder):
        shutil.rmtree(cropped_image_folder)
    if clear_existing and os.path.exists(cropped_label_folder):
        shutil.rmtree(cropped_label_folder)

    if not os.path.exists(cropped_image_folder):
        os.makedirs(cropped_image_folder)
    if not os.path.exists(cropped_label_folder):
        os.makedirs(cropped_label_folder)

    rois = get_rois(label_folder)

    for roi in tqdm(rois):
        crop_roi(roi, image_folder, label_folder, cropped_image_folder, cropped_label_folder,
                 padding, patch_size)


if __name__ == '__main__':
    params = parse_params("Run step 050 to crop out unannotated areas of the images.")

    images_raw_dir = '../'+params['scaled_images_dir']
    images_raw_labels_dir = '../'+params['scaled_labels_dir']
    cropped_images_dir = '../'+params['cropped_images_dir']
    cropped_labels_dir = '../'+params['cropped_labels_dir']

    padding = params['crop_padding']
    patch_size = params['model']['patch_shape']

    do_crops(images_raw_dir, images_raw_labels_dir, cropped_images_dir, cropped_labels_dir, patch_size[1:], padding)

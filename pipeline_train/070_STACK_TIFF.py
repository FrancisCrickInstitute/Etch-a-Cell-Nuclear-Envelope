"""
070_STACK_TIFF.py

This file combines individual images in to an image stack. The model accepts a patch of volume as
input, so it is convenient to be able to load stacks rather than individual images.
"""
import os
import shutil

import numpy as np
from skimage.io import imread, imsave
from tqdm import tqdm

from src.image_processing import get_stack_filenames, save_image, get_tag_resolution
from src.param_parser import parse_params


def create_tiff_stack_matching(source_images_dir, source_stacks_dir,
                               label_images_dir, label_stacks_dir, size_z, clear_existing=False, compress=False):
    if clear_existing and os.path.exists(source_stacks_dir):
        shutil.rmtree(source_stacks_dir)
    if clear_existing is True and os.path.exists(label_stacks_dir):
        shutil.rmtree(label_stacks_dir)
    if not os.path.exists(source_stacks_dir):
        os.makedirs(source_stacks_dir)
    if not os.path.exists(label_stacks_dir):
        os.makedirs(label_stacks_dir)

    stack_filenames, filenames, input_extension = get_stack_filenames(source_images_dir)
    image_range = []

    for stack_filename in tqdm(stack_filenames):
        image_stack = []
        label_stack = []
        source_image = []
        label_image = []
        mini = -1
        maxi = -1
        add_prefix_z = True
        resx_source = 1
        resy_source = 1
        res_unit_source = ""
        resx_label = 1
        resy_label = 1
        res_unit_label = ""

        for filename in filenames:
            file = filename.rsplit('_', 1)
            # find matching files
            if file[0] == stack_filename:
                slice = file[1]
                add_prefix_z = slice.lower().startswith("z")
                if add_prefix_z:
                    slice = slice[1:]
                i = int(slice)
                if i < mini or mini < 0:
                    mini = i
                if i > maxi:
                    maxi = i
                if len(source_image) == 0:
                    source_image = imread(source_images_dir + filename + input_extension)
                if len(label_image) == 0 and os.path.exists(label_images_dir + filename + input_extension):
                    label_image = imread(label_images_dir + filename + input_extension)

        image_range = [mini, maxi]

        for i in range(mini, maxi + 1):
            input_filename = stack_filename + '_'
            if add_prefix_z:
                input_filename += 'z'
            input_filename += f"{i:04d}" + input_extension

            input_path = source_images_dir + input_filename
            if os.path.exists(input_path):
                source_image = imread(input_path)
                resx_source, resy_source, _, res_unit_source = get_tag_resolution(input_path)
            else:
                source_image = np.zeros_like(source_image)
            image_stack.append(source_image)

            input_path = label_images_dir + input_filename
            if os.path.exists(input_path):
                label_image = imread(input_path)
                resx_label, resy_label, _, res_unit_label = get_tag_resolution(input_path)
            else:
                label_image = np.zeros_like(label_image)
            label_stack.append(label_image)

        np_stack = np.array(image_stack)
        if np_stack.dtype.kind == "O":
            print("Stack error", stack_filename, "Type:", np_stack.dtype, "Shape:", np_stack.shape)
        else:
            save_image(source_stacks_dir + stack_filename + '.tiff', np_stack, resx=resx_source, resy=resy_source, size_z=size_z,
                       res_unit=res_unit_source, compress=compress)

        np_stack = np.array(label_stack)
        if np_stack.dtype.kind == "O":
            print("Stack error", stack_filename, "Type:", np_stack.dtype, "Shape:", np_stack.shape)
        else:
            save_image(label_stacks_dir + stack_filename + '.tiff', np_stack, resx=resx_label, resy=resy_label, size_z=size_z,
                       res_unit=res_unit_label, compress=compress)

    return image_range


if __name__ == '__main__':
    params = parse_params("Run step 070 to stack tiff images.")

    #scaled_images_dir = '../'+params['scaled_images_dir']
    #scaled_labels_dir = '../'+params['scaled_labels_dir']
    #scaled_image_stacks_dir = '../'+params['scaled_image_stacks_dir']
    #scaled_label_stacks_dir = '../'+params['scaled_label_stacks_dir']

    #cropped_images_dir = '../'+params['cropped_images_dir']
    #cropped_labels_dir = '../'+params['cropped_labels_dir']
    #cropped_image_stacks_dir = '../'+params['cropped_image_stacks_dir']
    #cropped_label_stacks_dir = '../'+params['cropped_label_stacks_dir']

    #create_tiff_stack_matching(scaled_images_dir, scaled_image_stacks_dir, scaled_labels_dir, scaled_label_stacks_dir)
    #create_tiff_stack_matching(cropped_images_dir, cropped_image_stacks_dir, cropped_labels_dir, cropped_label_stacks_dir)

    images_dir = '../projects/nuclear/resources/images/raw/'
    images_stacks_dir = '../projects/nuclear/resources/images/raw-stacks2/'
    labels_dir = '../projects/nuclear/resources/images/raw-labels/'
    labels_stacks_dir = '../projects/nuclear/resources/images/raw-labels-stacks/'
    size_z = 0.05

    create_tiff_stack_matching(images_dir, images_stacks_dir, labels_dir, labels_stacks_dir, size_z)

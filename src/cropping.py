import os
import math
import glob

import numpy as np
from skimage.io import imread, imsave
from skimage.exposure import rescale_intensity


def crop_images_in_folder_for_roi(crop_shape, roi, read_folder, save_folder):
    filenames = glob.glob(read_folder+roi+'*')
    for filename in filenames:
        image = imread(filename)
        cropped_image = image[crop_shape[0]:crop_shape[1],
                              crop_shape[2]:crop_shape[3]]

        imsave(save_folder+os.path.basename(filename), cropped_image, check_contrast=False)


def get_annotation_mins_and_maxes_for_roi(roi, label_folder):
    """
    Get min and max locations of the segmentation per annotation, and group them by slice.
    """
    label_filenames = glob.glob(label_folder+roi+'*')

    label_groups = {}
    for label_filename in label_filenames:
        if '#' in label_filename:
            z_slice = os.path.basename(label_filename).split('_', 2)[2].rsplit('#', 1)[0]
        else:
            z_slice = os.path.basename(label_filename).split('_', 2)[2].rsplit('.', 1)[0]

        label = imread(label_filename)
        label = rescale_intensity(label, (0, 1))

        if z_slice not in label_groups:
            label_groups[z_slice] = [label]
        else:
            label_groups[z_slice].append(label)

    x_mins, x_maxes, y_mins, y_maxes = [], [], [], []
    for z_slice, label_group in label_groups.items():
        labels = np.array(label_group)
        locations = np.argwhere(labels != 0)

        if locations.shape[0] != 0:
            x_min, x_max = np.min(locations[:, 2]), np.max(locations[:, 2])
            y_min, y_max = np.min(locations[:, 1]), np.max(locations[:, 1])
            x_mins.append(x_min)
            x_maxes.append(x_max)
            y_mins.append(y_min)
            y_maxes.append(y_max)
    return x_mins, x_maxes, y_mins, y_maxes


def get_crop_shape(xy_mins_and_maxes, image_shape, padding, patch_size):
    x_min = int(np.median(xy_mins_and_maxes[0]))
    x_max = int(np.median(xy_mins_and_maxes[1]))
    y_min = int(np.median(xy_mins_and_maxes[2]))
    y_max = int(np.median(xy_mins_and_maxes[3]))

    # add padding, but not beyond original boundaries
    x_min -= min(padding, x_min)
    x_max += min(padding, image_shape[1] - x_max)
    y_min -= min(padding, y_min)
    y_max += min(padding, image_shape[0] - y_max)

    # make sure at least size of patch
    y_diff = y_max - y_min
    x_diff = x_max - x_min
    missing_y_pix = max(0, patch_size[0] - y_diff)
    missing_x_pix = max(0, patch_size[1] - x_diff)

    y_min -= int(missing_y_pix / 2)
    y_max += math.ceil(missing_y_pix / 2)
    x_min -= int(missing_x_pix / 2)
    x_max += math.ceil(missing_x_pix / 2)

    # correct if went outside boundaries
    if y_min < 0:
        y_max += min(image_shape[0] - y_max, abs(y_min))
        y_min = 0
    elif y_max > image_shape[0]:
        y_min -= min(y_max - image_shape[0], y_min)
        y_max = image_shape[0]
    if x_min < 0:
        x_max += min(image_shape[1] - x_max, abs(x_min))
        x_min = 0
    elif x_max > image_shape[1]:
        x_min -= min(x_max - image_shape[1], x_min)
        x_max = image_shape[1]

    return y_min, y_max, x_min, x_max


def get_image_shape(roi, image_folder):
    filenames = glob.glob(image_folder+roi+'*')
    image = imread(filenames[0])
    return image.shape


def crop_roi(roi, image_folder, label_folder, cropped_image_folder, cropped_label_folder,
             padding, patch_size):
    image_shape = get_image_shape(roi, image_folder)
    annotation_mins_and_maxes = get_annotation_mins_and_maxes_for_roi(roi, label_folder)
    crop = get_crop_shape(annotation_mins_and_maxes, image_shape, padding, patch_size)
    crop_images_in_folder_for_roi(crop, roi, image_folder, cropped_image_folder)
    crop_images_in_folder_for_roi(crop, roi, label_folder, cropped_label_folder)


def get_rois(label_folder):
    filenames = glob.glob(label_folder+'*')

    rois = []
    for filename in filenames:
        roi = os.path.basename(filename).rsplit('_', 1)[0]
        if roi not in rois:
            rois.append(roi)
    return rois



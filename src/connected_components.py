import os
import numpy as np
from skimage import measure
from skimage.io import imread
from tqdm import tqdm

from src.image_processing import to_binary, save_image, get_tag_resolution


def get_connected_components(image_stack):
    binary_stack = to_binary(image_stack)
    return measure.label(binary_stack, background=0)


def keep_regions_over_threshold(image_stack, threshold=10000):
    """ keep regions in the image stacks with at least the threshold of connected voxels """
    labelled_stack = get_connected_components(image_stack)
    intensities, occurrences = np.unique(labelled_stack, return_counts=True)
    for intensity in intensities:
        if occurrences[intensity] < threshold:
            labelled_stack[labelled_stack == intensity] = 0
    labelled_stack[labelled_stack != 0] = 1

    labelled_stack = labelled_stack.astype(np.uint8)*255
    return labelled_stack


def keep_largest_region(image_stack):
    """ keep the largest connected region inside an image stack """
    labelled_stack = get_connected_components(image_stack)
    intensities, occurrences = np.unique(labelled_stack, return_counts=True)
    # ignore first index, which is the background
    largest_idx = np.argmax(occurrences[1:])
    largest_intensity = intensities[1:][largest_idx]
    labelled_stack[labelled_stack != largest_intensity] = 0
    labelled_stack[labelled_stack == largest_intensity] = 1

    labelled_stack = labelled_stack.astype(np.uint8)*255
    return labelled_stack


if __name__ == '__main__':
    source_dir = "../projects/nuclear/resources/images/raw-labels-stacks"
    dest_dir = "../projects/nuclear/resources/images/raw-labels-stacks-cc"

    for file in tqdm(os.listdir(source_dir)):
        input_path = os.path.join(source_dir, file)
        output_path = os.path.join(dest_dir, file)

        source_stack = imread(input_path)
        resinfo = get_tag_resolution(input_path)

        dest_stack = keep_largest_region(source_stack)

        save_image(output_path, dest_stack,
                   resx=resinfo[0], resy=resinfo[1], size_z=resinfo[2], res_unit=resinfo[3],
                   compress=True)

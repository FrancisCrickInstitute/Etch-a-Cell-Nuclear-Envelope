"""
020_PREPROCESS_IMAGES

This file converts a tiff stack in to separate images. We aggregate annotations on a per image
basis, so it is convenient to be able to load just the reference image rather than a whole stack
containing it.
"""
import os

from src.image_processing import unstack_images
from src.param_parser import parse_params


def unstack(rawimage_dir, rawimagestack_dir, output_extension=".tiff", add_prefix_z=True, z_index_offset=0):
    if not os.path.exists(rawimage_dir):
        os.makedirs(rawimage_dir)

    nprocessed = unstack_images(rawimage_dir, rawimagestack_dir, output_extension, add_prefix_z, z_index_offset)

    print(f'Processed image stacks: {nprocessed}')
    print('Finished processing images...')


if __name__ == '__main__':
    params = parse_params("Run step 020 to unstack the source tiffs.")

    images_raw_dir = os.path.join('..', params['images_raw_dir'])
    images_raw_stack_dir = os.path.join('..', params['images_raw_stack_dir'])
    ref_image_z_offset = params['ref_images']['z_offset']

    unstack(images_raw_dir, images_raw_stack_dir, z_index_offset=ref_image_z_offset)

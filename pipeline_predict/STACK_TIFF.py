"""
STACK_TIFF.py
"""
import os
import shutil

from src.image_processing import stack_images
from src.param_parser import parse_params


def create_tiff_stack(images_dir, stacks_dir, size_z, clear_existing=False, compress=False):
    if clear_existing and os.path.exists(stacks_dir):
        shutil.rmtree(stacks_dir)
    if not os.path.exists(stacks_dir):
        os.makedirs(stacks_dir)

    return stack_images(images_dir, stacks_dir, size_z, compress=compress)


if __name__ == '__main__':
    params = parse_params("Run to stack tiff images.")

    #images_dir = os.path.join('..', params['images_raw_dir'])
    #stacks_dir = os.path.join('..', params['images_raw_stack_dir'])

    stacks_dir = '../projects/nuclear/resources/images/raw-stacks2'
    images_dir = '../projects/nuclear/resources/images/raw'
    size_z = 0.05

#    images_dir = '../projects/nuclear/resources/images/raw-labels'
#    stacks_dir = '../projects/nuclear/resources/images/raw-labels-stacks-complete'
    create_tiff_stack(images_dir, stacks_dir, size_z)

"""
UNSTACK_TIFF.py
"""
import os
import shutil

from src.image_processing import unstack_images
from src.param_parser import parse_params


def unstack_tiff(images_dir, stacks_dir, clear_existing=False, compress=False):
    if clear_existing and os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    return unstack_images(images_dir, stacks_dir, compress=compress)


if __name__ == '__main__':
    params = parse_params("Run to unstack tiff images.")

    stacks_dir = '../projects/nuclear/resources/images/raw-stacks'
    images_dir = '../projects/nuclear/resources/images/raw'
    unstack_tiff(images_dir, stacks_dir)

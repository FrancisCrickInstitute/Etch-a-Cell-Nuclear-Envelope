"""
CONNECTED.py

Run connected components analysis to get rid of small bits of oversegmentation.
"""
import os

from skimage.io import imread
from tqdm import tqdm

from src.connected_components import keep_regions_over_threshold
from src.image_processing import save_image
from src.helpers import sizenm_to_dpum


def remove_small_regions(images_dir, model_size_xy_nm, model_size_z_nm):
    resxy = sizenm_to_dpum(model_size_xy_nm)
    size_z_um = model_size_z_nm / 1000
    res_unit = "micron"

    for image_stack_filename in tqdm(os.listdir(images_dir)):
        image_file_path = os.path.join(images_dir, image_stack_filename)
        image_stack = imread(image_file_path)
        image_stack = keep_regions_over_threshold(image_stack)

        save_image(image_file_path, image_stack,
                   resx=resxy, resy=resxy, size_z=size_z_um, res_unit=res_unit, compress=True)


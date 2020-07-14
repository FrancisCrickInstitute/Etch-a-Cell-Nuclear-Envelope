"""
040_DOWNSAMPLE.py

This file downscales image data. In order that the machine learning model can see a reasonable
portion of the cell at one time, while still fitting in memory, we pass it image data which
is of a lower resolution than is available.
"""
import os
import shutil
from tqdm import tqdm

from src.helpers import load_processed_csv
from src.image_processing import scale_save_image
from src.param_parser import parse_params


def get_xy_scale(csv_filename, targetsize_nm_xy):
    # assumption: all annotations for the same slice have the same physical resolution... this should be the case

    if not os.path.exists(csv_filename):
        # csv file missing for this slice, find nearest
        filename, ext = os.path.splitext(csv_filename)
        filebase, offset = filename.rsplit('_', 1)
        add_prefix_z = (offset.startswith("z"))
        if add_prefix_z:
            offset = offset[1:]
        index = int(offset)
        delta = 1
        tries = 0
        while not os.path.exists(csv_filename) and tries < 1000:
            index += delta
            csv_filename = filebase + "_"
            if add_prefix_z:
                csv_filename += "z"
            csv_filename += f"{abs(index):04d}" + ext
            if delta < 0:
                delta = -delta + 1
            else:
                delta = -(delta + 1)
            tries += 1

        if tries >= 1000:
            return 0

    csv_file = load_processed_csv(csv_filename)

    # take most frequent reference value (there are occasional errors in the csv)
    res_nm_xy = csv_file['raw xy resolution (nm)'].mode()[0]
    scale_xy = res_nm_xy / targetsize_nm_xy

    # TODO: if z scale is not target z scale, 'skip' or interpolate images somehow?
    # res_nm_z = csv_file['raw z resolution (nm)'][0]
    # scale_z = res_nm_z / targetsize_nm_z

    return scale_xy


def rescale(csv_dir, raw_dir, scaled_dir, targetsize_nm_xy, targetsize_nm_z, binary_format=False, clear_existing=False):
    if clear_existing and os.path.exists(scaled_dir):
        shutil.rmtree(scaled_dir)
    if not os.path.exists(scaled_dir):
        os.makedirs(scaled_dir)

    for file in tqdm(os.listdir(raw_dir)):
        filename, ext = os.path.splitext(file)
        csv_filename = csv_dir + filename + '.csv'
        scale_xy = get_xy_scale(csv_filename, targetsize_nm_xy)
        if scale_xy != 0:
            scale_save_image(raw_dir, scaled_dir, filename, scale_xy, targetsize_nm_z, binary_format=binary_format)


if __name__ == '__main__':
    params = parse_params("Run step 040 to downscale the images and labels.")

    processed_csv_dir = os.path.join('..', params['processed_csv_dir'])
    images_raw_dir = os.path.join('..', params['images_raw_dir'])
    images_raw_labels_dir = os.path.join('..', params['images_raw_labels_dir'])
    scaled_images_dir = os.path.join('..', params['scaled_images_dir'])
    scaled_labels_dir = os.path.join('..', params['scaled_labels_dir'])

    target_xy_nm = params['target_xy_nm']
    target_z_nm = params['target_z_nm']

    print('Downscaling source images')
    rescale(processed_csv_dir, images_raw_dir, scaled_images_dir, target_xy_nm, target_z_nm, binary_format=False)
    print('Downscaling label images')
    rescale(processed_csv_dir, images_raw_labels_dir, scaled_labels_dir, target_xy_nm, target_z_nm, binary_format=True)



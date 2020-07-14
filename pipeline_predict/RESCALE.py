"""
RESCALE.py
"""
import os
import shutil
import json
import argparse
from tqdm import tqdm

from src.helpers import get_file, dpum_to_sizenm
from src.image_processing import scale_save_image, get_tag_resolution, get_tag_imagesize


def rescale(source_dir, target_dir, ref_dir, sourcesize_nm_xy0, targetsize_nm_xy0, targetsize_nm_z, clear_existing=False, binary_format=False, compress=False):
    if clear_existing and os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for file in tqdm(os.listdir(source_dir)):
        filename, ext = os.path.splitext(file)
        sourcesize_nm_xy = sourcesize_nm_xy0
        targetsize_nm_xy = targetsize_nm_xy0

        # use source image resolution (assume pixes / micron)
        if not sourcesize_nm_xy:
            filename_source = os.path.join(source_dir, file)
            resx, resy, size_z, res_unit = get_tag_resolution(filename_source)
            if resx:
                sourcesize_nm_xy = dpum_to_sizenm(resx)
            else:
                print("Rescale error: no source resolution for: " + filename_source)

        if not targetsize_nm_xy:
            filename_source = os.path.join(source_dir, file)
            swidth, sheight = get_tag_imagesize(filename_source)
            filename_dest = os.path.join(ref_dir, filename) + ".*"
            dwidth, dheight = get_tag_imagesize(get_file(filename_dest))
            if swidth and dwidth:
                targetsize_nm_xy = sourcesize_nm_xy * (swidth / dwidth + sheight / dheight) / 2
            else:
                print("Rescale error: no destination size for: " + filename_dest)

        if sourcesize_nm_xy and targetsize_nm_xy:
            scale_xy = sourcesize_nm_xy / targetsize_nm_xy
        else:
            scale_xy = 1

        # TODO: if z scale is not target z scale, resample images in z direction
        # scale_z = sourcesize_nm_z / targetsize_nm_z
        scale_z = 1

        if scale_xy:
            scale_save_image(source_dir, target_dir, filename, scale_xy, scale_z, binary_format=binary_format, compress=compress)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run to rescale the images.")
    parser.add_argument('--params',
                        help='The location of the parameters file.',
                        default='../projects/nuclear/nuclear.json')
    parser.add_argument('--source',
                        help='The location of the source images.',
                        default='projects/nuclear/resources/images/raw-stacks')
    parser.add_argument('--scaled',
                        help='The location of the scaled images.',
                        default='projects/nuclear/resources/images/scaled-stacks')
    parser.add_argument('--ref',
                        help='The location of the reference images for the target resolution.',
                        default='projects/nuclear/resources/images/raw-stacks')
    parser.add_argument('--resxy',
                        help='The XY resolution of the source images [nm].')

    raw_dir = os.path.join("..", parser.parse_args().source)
    scaled_dir = os.path.join("..", parser.parse_args().scaled)
    ref_dir = os.path.join("..", parser.parse_args().ref)

    with open(parser.parse_args().params, 'r') as f:
        params = json.load(f)

    targetsize_xy_nm = params['target_xy_nm']
    target_z_nm = params['target_z_nm']

    #print('Downscaling source images')
    #rescale(raw_dir, scaled_dir, ref_dir, sourcesize_nm_xy, targetsize_xy_nm, target_z_nm)

    rescale("../projects/nuclear/resources/images/scaled-predictions-stacks-cc",
            "../projects/nuclear/resources/images/predictions-stacks-cc",
            "../projects/nuclear/resources/images/raw-stacks",
            targetsize_xy_nm, 0, target_z_nm, compress=True)

"""
030_AGGREGATE.py

This file performs aggregation of the citizen science data. Each cell image has been presented to
many different volunteers, and one of the most important pre-processing actions to perform is taking
a collection of citizen scientist annotations for a single slice and finding some type of consensus
as to where in the image the feature of interest (for example nuclear membrane) actually is.
"""
import os
import numpy as np
from skimage.io import imread
from tqdm import tqdm
import shutil

from src.helpers import load_processed_csv, dpum_to_sizenm
from src.image_processing import get_tag_resolution, save_image
from src.interiors_probability import do_interiors_contours, get_annotation_points
from src.param_parser import parse_params
from src.helpers import get_file


def save_aggregations(output_filepath, aggregations, res_info):
    """
    Convert image matrix to 32 bit and save.
    """
    if type(aggregations) is list:
        for i, aggregation in enumerate(aggregations):
            filepath, ext = os.path.splitext(output_filepath)
            savepath = filepath+'#'+str(i)+ext

            aggregation = aggregation.astype(np.float32)
            save_image(savepath, aggregation, resx=res_info[0], resy=res_info[1], size_z=res_info[2], res_unit=res_info[3],
                       compress=True)
    else:
        aggregations = aggregations.astype(np.float32)
        save_image(output_filepath, aggregations, resx=res_info[0], resy=res_info[1], size_z=res_info[2], res_unit=res_info[3],
                   compress=True)


def aggregate(csv_dir, ref_images_dir, output_dir, border_width_nm, output_extension='.tiff',
              method='probability', clear_existing=False, zoom_factor=1, correct_width=2000, correct_height=2000):
    if clear_existing and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    missing_ref_images = 0
    for csv in tqdm(os.listdir(csv_dir)):
        data_frame = load_processed_csv(csv_dir + csv)
        filename, _ = os.path.splitext(csv)
        input_filepath = get_file(os.path.join(ref_images_dir, filename+".*"))
        output_filepath = os.path.join(output_dir, filename + output_extension)
        # avoid having to redo aggregations that are already done - delete dir if really need to restart
        if not os.path.exists(output_filepath):
            if not input_filepath:
                missing_ref_images += 1
            else:
                ref_image = imread(input_filepath)
                height, width = ref_image.shape

                res_info = get_tag_resolution(input_filepath)
                size_nm = dpum_to_sizenm(res_info[0])
                border_width = round(border_width_nm / size_nm)

                if height != correct_height or width != correct_width:
                    #print(f"Correcting image {input_filepath} width/height: {width}/{height}")
                    zoom_factor_x = zoom_factor * width / correct_width
                    zoom_factor_y = zoom_factor * height / correct_height
                else:
                    zoom_factor_x = zoom_factor
                    zoom_factor_y = zoom_factor

                # input: 'data_frame': csv annotations
                # output: 'aggregation': (image) matrix
                if method == 'interiors-contours':
                    annotations = get_annotation_points(data_frame, zoom_factor_x, zoom_factor_y)
                    aggregation = do_interiors_contours(annotations,
                                                        width=width,
                                                        height=height,
                                                        border_width=border_width)
                else:
                    raise ValueError(f'Invalid aggregation method: \'{method}\'.')

                #illustrate_draw_annotations(output_dir + "/..", annotations, width, height, border_width, True)
                #illustrate_area_annotations(output_dir + "/..", annotations, width, height)

                save_aggregations(output_filepath, aggregation, res_info)
    print(f"Aggregation failures because of a missing reference image: {missing_ref_images}")


if __name__ == '__main__':
    params = parse_params("Run step 030 to aggregate citizen science annotations.")

    csv_dir = os.path.join('..', params['processed_csv_dir'])
    ref_images_dir = os.path.join('..', params['images_raw_dir'])
    labels_dir = os.path.join('..', params['images_raw_labels_dir'])

    ref_image_zoom = params['ref_images']['zoom_factor']
    ref_image_target_width = params['ref_images']['target_width']
    ref_image_target_height = params['ref_images']['target_height']

    aggregation_method = params['aggregation_method']
    border_width_nm = params['border_width_nm']

    aggregate(csv_dir, ref_images_dir, labels_dir, border_width_nm=border_width_nm, method=aggregation_method,
              zoom_factor=ref_image_zoom, correct_width=ref_image_target_width, correct_height=ref_image_target_height)

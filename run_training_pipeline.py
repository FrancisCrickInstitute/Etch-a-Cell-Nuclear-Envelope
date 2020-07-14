import json
import argparse

from importlib import import_module


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run the training pipeline.")
    parser.add_argument('--params',
                        help='The location of the parameters file.',
                        default='projects/nuclear/nuclear.json')
    parser.add_argument('--ignore_steps',
                        nargs='*',
                        type=int,
                        help='Ignore specified pipeline steps.',
                        default=[])
    parser.add_argument('--restart',
                        type=bool,
                        help='Clears previous pre-processing data while running the pipeline.',
                        default=True)

    args = parser.parse_args()
    ignore_steps = args.ignore_steps
    restart = args.restart
    with open(args.params, 'r') as f:
        params = json.load(f)

    PREPROCESS = import_module('pipeline_train.010_PREPROCESS_CSV')
    UNSTACK = import_module('pipeline_train.020_PREPROCESS_IMAGES')
    AGGREGATE = import_module('pipeline_train.030_AGGREGATE')
    DOWNSAMPLE = import_module('pipeline_train.040_DOWNSAMPLE')
    CROP = import_module('pipeline_train.050_CROP')
    DISCARD = import_module('pipeline_train.060_DISCARD_FOR_TRAINING')
    STACKTIFF = import_module('pipeline_train.070_STACK_TIFF')
    TRAIN = import_module('pipeline_train.080_TRAIN')

    zooniverse_csv_file = params['zooniverse_csv_file']
    processed_csv_dir = params['processed_csv_dir']

    images_raw_dir = params['images_raw_dir']
    images_raw_stack_dir = params['images_raw_stack_dir']
    images_raw_labels_dir = params['images_raw_labels_dir']

    scaled_images_dir = params['scaled_images_dir']
    scaled_labels_dir = params['scaled_labels_dir']
    scaled_image_stacks_dir = params['scaled_image_stacks_dir']
    scaled_label_stacks_dir = params['scaled_label_stacks_dir']

    cropped_images_dir = params['cropped_images_dir']
    cropped_labels_dir = params['cropped_labels_dir']
    cropped_image_stacks_dir = params['cropped_image_stacks_dir']
    cropped_label_stacks_dir = params['cropped_label_stacks_dir']

    model_images_dir = params['data']['images_dir']
    model_labels_dir = params['data']['labels_dir']
    model_save_dir = params['model']['save_dir']

    zooniverse_workflow = params['zooniverse_workflow']
    aggregation_method = params['aggregation_method']
    border_width_nm = params['border_width_nm']

    target_xy_nm = params['target_xy_nm']
    target_z_nm = params['target_z_nm']
    size_z_um = target_z_nm / 1000

    ref_image_z_offset = params['ref_images']['z_offset']
    ref_image_zoom = params['ref_images']['zoom_factor']
    ref_image_target_width = params['ref_images']['target_width']
    ref_image_target_height = params['ref_images']['target_height']

    padding = params['crop_padding']
    patch_size = params['model']['patch_shape']

    print('\n=====> PIPELINE STEP 1/8 ---  Zooniverse CSV format conversion')
    if 1 not in ignore_steps:
        PREPROCESS.preprocess_zooniverse_csv(output_dir=processed_csv_dir, input_path=zooniverse_csv_file,
                                             workflow=zooniverse_workflow)
    else:
        print('...SKIPPED...')

    print('\n=====> PIPELINE STEP 2/8 --- Unstacking reference images')
    if 2 not in ignore_steps:
        UNSTACK.unstack(images_raw_dir, images_raw_stack_dir, z_index_offset=ref_image_z_offset)
    else:
        print('...SKIPPED...')

    print('\n=====> PIPELINE STEP 3/8 --- Aggregating the annotations')
    if 3 not in ignore_steps:
        AGGREGATE.aggregate(processed_csv_dir, images_raw_dir, images_raw_labels_dir, border_width_nm=border_width_nm,
                            method=aggregation_method, clear_existing=restart, zoom_factor=ref_image_zoom,
                            correct_width=ref_image_target_width, correct_height=ref_image_target_height)
    else:
        print('...SKIPPED...')

    print('\n=====> PIPELINE STEP 4/8 --- Downscale labels and reference images')
    if 4 not in ignore_steps:
        print('Downscaling source images')
        DOWNSAMPLE.rescale(processed_csv_dir, images_raw_dir, scaled_images_dir, target_xy_nm, target_z_nm,
                           binary_format=False, clear_existing=restart)
        print('Downscaling label images')
        DOWNSAMPLE.rescale(processed_csv_dir, images_raw_labels_dir, scaled_labels_dir, target_xy_nm, target_z_nm,
                           binary_format=True, clear_existing=restart)
    else:
        print('...SKIPPED...')

    print('\n=====> PIPELINE STEP 5/8 --- Cropping out annotation free regions')
    if 5 not in ignore_steps:
        CROP.do_crops(scaled_images_dir, scaled_labels_dir, cropped_images_dir, cropped_labels_dir, patch_size[1:],
                      padding, clear_existing=restart)
    else:
        print('...SKIPPED...')

    print('\n=====> PIPELINE STEP 6/8 --- Discarding slices with too few annotations')
    if 6 not in ignore_steps:
        DISCARD.discard_ref_images(processed_csv_dir, cropped_images_dir, cropped_labels_dir)
    else:
        print('...SKIPPED...')

    print('\n=====> PIPELINE STEP 7/8 --- Creating tiff stacks')
    if 7 not in ignore_steps:
        STACKTIFF.create_tiff_stack_matching(scaled_images_dir, scaled_image_stacks_dir,
                                             scaled_labels_dir, scaled_label_stacks_dir, size_z_um, clear_existing=restart)
        STACKTIFF.create_tiff_stack_matching(cropped_images_dir, cropped_image_stacks_dir,
                                             cropped_labels_dir, cropped_label_stacks_dir, size_z_um, clear_existing=restart)
    else:
        print('...SKIPPED...')

    print('\n=====> PIPELINE STEP 8/8 --- Training machine learning model')
    if 8 not in ignore_steps:
        TRAIN.train_model(params, model_images_dir, model_labels_dir, model_save_dir, aggregation_method)
    else:
        print('...SKIPPED...')

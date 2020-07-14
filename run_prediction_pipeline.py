import argparse
import json

from pipeline_predict import PREDICT, RESCALE, STACK_TIFF, CONNECTED


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run the prediction pipeline.")
    parser.add_argument('--params',
                        help='The location of the parameters file.',
                        default='projects/nuclear/nuclear.json')
    parser.add_argument('--model',
                        help='Explicitly pass the model path/name to use for prediction.',
                        default=None)
    parser.add_argument('--ignore_steps',
                        nargs='*',
                        type=int,
                        help='Ignore specified pipeline steps.',
                        default=[])
    parser.add_argument('--restart',
                        type=bool,
                        help='Clears previous data while running the pipeline.',
                        default=True)
    parser.add_argument('--source',
                        help='The location of the source images.',
                        default='projects/nuclear/resources/images/raw-stacks')
    parser.add_argument('--predictions',
                        help='The location of the predictions output images.',
                        default='projects/nuclear/resources/images/predictions-stacks')
    parser.add_argument('--tri_axis',
                        help='Run predictions across all 3 axes (x, y, z) and then recombine for the final prediction.',
                        default=False)

    args = parser.parse_args()
    ignore_steps = args.ignore_steps
    restart = args.restart
    tri_axis = args.tri_axis
    with open(args.params, 'r') as f:
        params = json.load(f)

    source_dir = args.source
    predictions_dir = args.predictions
    model_name = args.model

    stack_dir = "projects/nuclear/resources/images/raw-stacks"
    scaled_dir = "projects/nuclear/resources/images/scaled-stacks"
    scaled_predictions_dir = "projects/nuclear/resources/images/scaled-predictions-stacks"

    model_dir = params['model']['save_dir']
    aggregation_method = params['aggregation_method']
    target_xy_nm = params['target_xy_nm']
    target_z_nm = params['target_z_nm']
    size_z_um = target_z_nm / 1000

    print('\n=====> PIPELINE STEP 1/5 --- Stacking source images')
    if 1 not in ignore_steps and "stack" not in source_dir:
        STACK_TIFF.create_tiff_stack(source_dir, stack_dir, size_z_um, clear_existing=restart)
    else:
        if "stack" in source_dir:
            stack_dir = source_dir
        print('...SKIPPED...')

    print('\n=====> PIPELINE STEP 2/5 --- Downscaling source images')
    if 2 not in ignore_steps:
        RESCALE.rescale(stack_dir, scaled_dir, "", 0, target_xy_nm, target_z_nm, clear_existing=restart)
    else:
        print('...SKIPPED...')

    print('\n=====> PIPELINE STEP 3/5 --- Evaluating images on model')
    if 3 not in ignore_steps:
        PREDICT.predict(params, scaled_dir, scaled_predictions_dir, model_dir, aggregation_method, target_xy_nm, target_z_nm, model_name=model_name, clear_existing=restart, tri_axis=tri_axis)
    else:
        print('...SKIPPED...')

    print('\n=====> PIPELINE STEP 4/5 --- Removing oversegmentation artefacts')
    if 4 not in ignore_steps:
        CONNECTED.remove_small_regions(scaled_predictions_dir, target_xy_nm, target_z_nm)
    else:
        print('...SKIPPED...')

    print('\n=====> PIPELINE STEP 5/5 --- Upscaling label images')
    if 5 not in ignore_steps:
        RESCALE.rescale(scaled_predictions_dir, predictions_dir, stack_dir, target_xy_nm, 0, target_z_nm, clear_existing=restart, binary_format=True, compress=True)
    else:
        print('...SKIPPED...')

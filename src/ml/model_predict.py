import os
import time

import numpy as np
from skimage.io import imsave, imread
from tqdm import tqdm

from src.helpers import sizenm_to_dpum
from src.image_processing import save_image, to_binary
from src.ml.DataLoader import get_image_patch
from src.ml.augmentation import normalize_batch


def evaluate(model, volume):
    """
    :model: a keras model object
    :volume: a 3D image volume (ndarray) to obtain the 3d segmentation map for
    """

    PATCH_SHAPE = model.input_shape[:0:-1]

    OVERLAP_X_Y = PATCH_SHAPE[1] // 4
    OVERLAP_Z = PATCH_SHAPE[0] // 4

    # padding guarantees that the whole volume is convolved over.
    padded_volume = np.pad(
        volume,
        (
            (OVERLAP_Z + PATCH_SHAPE[0], OVERLAP_Z + PATCH_SHAPE[0]),
            (OVERLAP_X_Y + PATCH_SHAPE[1], OVERLAP_X_Y + PATCH_SHAPE[1]),
            (OVERLAP_X_Y + PATCH_SHAPE[2], OVERLAP_X_Y + PATCH_SHAPE[2])
        ),
        'symmetric'
    )

    D, H, W = padded_volume.shape
    #print(padded_volume.shape)

    grid_coordinates = [
        (z, y, x)
        for z in range(0, D - PATCH_SHAPE[0], PATCH_SHAPE[0] - (OVERLAP_Z * 2))
        for y in range(0, H - PATCH_SHAPE[1], PATCH_SHAPE[1] - (OVERLAP_X_Y * 2))
        for x in range(0, W - PATCH_SHAPE[2], PATCH_SHAPE[2] - (OVERLAP_X_Y * 2))
    ]

    result_volume = np.zeros_like(padded_volume, dtype=np.float32)
    # split up computation into batches because thats a lot of patches
    #print(f'Running prediction over {len(grid_coordinates)} patches')

    # make divisible by 8 for tpu
    tpu_coords = [grid_coordinates[x:x + 128] for x in range(0, len(grid_coordinates) - 128, 128)]

    for batch_coordinates in tpu_coords:

        image_patches = []
        for corner in batch_coordinates:
            image_patch = get_image_patch(padded_volume, PATCH_SHAPE, corner)
            image_patches.append(image_patch)

        image_patches = np.array(image_patches)
        image_patches = np.moveaxis(image_patches, 1, 3)  # tf --> (N, W, H, D)
        normalized_image_patches = normalize_batch(image_patches)

        predictions = model.predict(normalized_image_patches)

        """
        print(predictions.shape)
        predictions = predictions[0]
        predictions = np.moveaxis(predictions, 2, 0)
        print(predictions.shape)
        plt.figure(figsize=(20, 20))
        plt.imshow(predictions[0]*255, cmap='gray', vmin=0, vmax=255)
        plt.show()
        return
        """

        for i, (d, h, w) in enumerate(batch_coordinates):
            # crop out the patch overlap (remove the perimiter)
            p = np.moveaxis(predictions[i], 2, 0)[
                OVERLAP_Z: -OVERLAP_Z,
                OVERLAP_X_Y: -OVERLAP_X_Y,
                OVERLAP_X_Y: -OVERLAP_X_Y
                ]
            # insert that crop into the right place in the result image
            result_volume[
            d + OVERLAP_Z: d + PATCH_SHAPE[0] - OVERLAP_Z,
            h + OVERLAP_X_Y: h + PATCH_SHAPE[1] - OVERLAP_X_Y,
            w + OVERLAP_X_Y: w + PATCH_SHAPE[2] - OVERLAP_X_Y
            ] = p

    # handle the missing patches from TPU divisible batches

    missing = len(grid_coordinates) % 128
    extra_batch = grid_coordinates[len(grid_coordinates) - missing:]
    extra_batch += [(0, 0, 0)] * (128 - missing)

    image_patches = []
    for corner in extra_batch:
        image_patch = get_image_patch(padded_volume, PATCH_SHAPE, corner)
        image_patches.append(image_patch)

    image_patches = np.array(image_patches)
    image_patches = np.moveaxis(image_patches, 1, 3)  # tf --> (N, W, H, D)
    normalized_image_patches = normalize_batch(image_patches)

    predictions = model.predict(normalized_image_patches)

    for i, (d, h, w) in enumerate(extra_batch[:missing]):
        # crop out the patch overlap (remove the perimiter)
        p = np.moveaxis(predictions[i], 2, 0)[
            OVERLAP_Z: -OVERLAP_Z,
            OVERLAP_X_Y: -OVERLAP_X_Y,
            OVERLAP_X_Y: -OVERLAP_X_Y
            ]
        # insert that crop into the right place in the result image
        result_volume[
        d + OVERLAP_Z: d + PATCH_SHAPE[0] - OVERLAP_Z,
        h + OVERLAP_X_Y: h + PATCH_SHAPE[1] - OVERLAP_X_Y,
        w + OVERLAP_X_Y: w + PATCH_SHAPE[2] - OVERLAP_X_Y
        ] = p

    # remove the padding to restore original shape
    result_volume = result_volume[
                    OVERLAP_Z + PATCH_SHAPE[0]: -(OVERLAP_Z + PATCH_SHAPE[0]),
                    OVERLAP_X_Y + PATCH_SHAPE[1]: -(OVERLAP_X_Y + PATCH_SHAPE[1]),
                    OVERLAP_X_Y + PATCH_SHAPE[2]: -(OVERLAP_X_Y + PATCH_SHAPE[2]),
                    ]

    return result_volume


def model_predict(model, source_dir, predictions_dir, model_size_xy_nm, model_size_z_nm, tri_axis=False):
    resxy = sizenm_to_dpum(model_size_xy_nm)
    size_z_um = model_size_z_nm / 1000
    res_unit = "micron"
    for filename in tqdm(os.listdir(source_dir)):
        input_path = os.path.join(source_dir, filename)
        input_image = imread(input_path)

        output_label = evaluate(model, input_image)
        if tri_axis:
            z_depth = input_image.shape[0]
            # determine whether or not z axis deep enough, mirror along z if not
            if z_depth < model.input_shape[:0:-1][1]:
                print(f'\n{filename}: z axis depth too shallow for 3-axis prediction, mirroring stack')
                input_image = np.concatenate((input_image, input_image[::-1, :, :]), axis=0)

            x_switch_label = np.swapaxes(evaluate(model, np.swapaxes(input_image, 0, 2)), 0, 2)
            y_switch_label = np.swapaxes(evaluate(model, np.swapaxes(input_image, 0, 1)), 0, 1)

            split_output_dir = os.path.join(predictions_dir, '../', 'xyz-split-predictions')
            if not os.path.exists(split_output_dir):
                os.makedirs(split_output_dir)

            # if we mirror to get a large enough z-axis, drop the extra slices after transposing back
            if z_depth < model.input_shape[:0:-1][1]:
                x_switch_label = x_switch_label[:z_depth, :, :]
                y_switch_label = y_switch_label[:z_depth, :, :]

            z_output_path = os.path.join(split_output_dir, 'z_'+filename)
            save_image(z_output_path, output_label, resx=resxy, resy=resxy, size_z=size_z_um, res_unit=res_unit, compress=True)
            x_output_path = os.path.join(split_output_dir, 'x_'+filename)
            save_image(x_output_path, x_switch_label, resx=resxy, resy=resxy, size_z=size_z_um, res_unit=res_unit, compress=True)
            y_output_path = os.path.join(split_output_dir, 'y_'+filename)
            save_image(y_output_path, y_switch_label, resx=resxy, resy=resxy, size_z=size_z_um, res_unit=res_unit, compress=True)

            # gather and threshold
            #output_label = output_label/3 + x_switch_label/3 + y_switch_label/3  # averaging version
            output_label = to_binary(output_label)
            x_switch_label = to_binary(x_switch_label)
            y_switch_label = to_binary(y_switch_label)
            output_label = np.logical_or(output_label, np.logical_or(x_switch_label, y_switch_label))*1.
            output_label = to_binary(output_label)

        output_path = os.path.join(predictions_dir, filename)
        save_image(output_path, output_label, resx=resxy, resy=resxy, size_z=size_z_um, res_unit=res_unit, compress=True)

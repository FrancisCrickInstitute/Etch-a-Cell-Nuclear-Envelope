"""
060_DISCARD_FOR_TRAINING.py

This file discards from training data those images where there were few or no annotations available.
We discard those with only a small number of annotations because without a significant number, a
rogue annotation can create a very poor training example.
"""
import os

from tqdm import tqdm

from src.helpers import get_file, load_processed_csv
from src.param_parser import parse_params


def check_annotations(csv_dir, ref_images_dir, labels_dir, min_annotations=5):
    non_labelled = 0
    # simple file discard
    for filename in tqdm(os.listdir(ref_images_dir)):
        label_filepath = labels_dir + filename
        if not os.path.exists(label_filepath):
            non_labelled += 1

    print(f"Non labelled source images: {non_labelled}")

    low_annotations = 0
    # check number of annotations
    for csv in tqdm(os.listdir(csv_dir)):
        data_frame = load_processed_csv(csv_dir + csv)
        if len(data_frame) < min_annotations:
            low_annotations += 1

    print(f"images with low annotations: {low_annotations}")


def discard_ref_images(csv_dir, ref_images_dir, labels_dir, min_annotations=5):
    ndiscarded = 0
    # simple file discard
    print("Discarding non labelled source images")
    for filename in tqdm(os.listdir(ref_images_dir)):
        image_filepath = ref_images_dir + filename
        label_filepath = labels_dir + filename
        if not os.path.exists(label_filepath):
            os.remove(image_filepath)
            ndiscarded += 1

    print(f'Discarded images: {ndiscarded}')

    ndiscarded = 0
    # check number of annotations
    print("Discarding images with low annotations")
    for csv in tqdm(os.listdir(csv_dir)):
        data_frame = load_processed_csv(csv_dir + csv)
        filename, _ = os.path.splitext(csv)
        image_filepath = get_file(ref_images_dir + filename + ".*")
        label_filepath = get_file(labels_dir + filename + ".*")

        if len(data_frame) < min_annotations:
            if os.path.exists(image_filepath):
                os.remove(image_filepath)
            if os.path.exists(label_filepath):
                os.remove(label_filepath)
            ndiscarded += 1

    print(f'Discarded images: {ndiscarded}')


if __name__ == '__main__':
    params = parse_params("Run step 060 to remove slices with very few annotations.")

    processed_csv_dir = os.path.join('..', params['processed_csv_dir'])
    cropped_images_dir = os.path.join('..', params['cropped_images_dir'])
    cropped_labels_dir = os.path.join('..', params['cropped_labels_dir'])

    #check_annotations(processed_csv_dir, cropped_images_dir, cropped_labels_dir)
    #discard_ref_images(processed_csv_dir, cropped_images_dir, cropped_labels_dir)
    discard_ref_images(processed_csv_dir, "../projects/nuclear/resources/images/raw/", "../projects/nuclear/resources/images/raw-labels/")

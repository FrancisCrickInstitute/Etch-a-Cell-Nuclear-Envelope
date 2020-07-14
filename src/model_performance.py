import os
import argparse
from skimage.io import imread

from src.helpers import get_file
from src.image_processing import to_binary
from src.performance_metrics import average_hausdorff_dist, fmeasure_area, fmeasure, recall_precision, \
    illustrate_area_images, mcc_area, boundary_to_interior_2d

import matplotlib.pyplot as plt


def model_performance(label_dir, predict_dir):
    print(f"Label (truth) folder: {label_dir}")
    print(f"Predictions folder: {predict_dir}")
    for stack_filename in os.listdir(label_dir):
        filebase, ext = os.path.splitext(stack_filename)
        label_image = to_binary(imread(os.path.join(label_dir, stack_filename)))
        predict_filename = get_file(os.path.join(predict_dir, filebase) + ".*")
        if predict_filename:
            predict_image = to_binary(imread(predict_filename))
            imagei = int((len(label_image) - 1) / 2)

            print(filebase)
            print("Average Hausdorff distance =", average_hausdorff_dist(label_image, predict_image))
            print("Boundary F-measure, recall, precision =", fmeasure(label_image, predict_image), recall_precision(label_image, predict_image))
            print(f"Image: {imagei}")
            print("Area average F-measure =", fmeasure_area(label_image[imagei], predict_image[imagei]))
            #plt.imshow(boundary_to_interior_2d(predict_image[imagei]))
            #plt.show()
            #print("Area average MCC =", mcc_area(label_image[imagei], predict_image[imagei]))

            #illustrate_area_images(label_dir + "/../" + filebase, label_image[imagei], predict_image[imagei])

        else:
            print("Matching prediction file not found: ", stack_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Model testing")
    parser.add_argument('--labels',
                        help='The location of the source images.',
                        default='projects/nuclear/resources/images/raw-labels-stacks-complete-cc')
    parser.add_argument('--predictions',
                        help='The location of the output label images.',
                        default='projects/nuclear/resources/images/predictions-stacks-cc')

    args = parser.parse_args()
    label_dir = os.path.join("..", args.labels)
    predict_dir = os.path.join("..", args.predictions)

    model_performance(label_dir, predict_dir)

"""
  Hausdorff distance is a well established measure of similarity between two lines.
  Our lines are not vectors, but pixels. Furthermore, they are probabilistic.
  These two constraints mean that the standard Hausdorff distance had to be adapted for purpose,
  such that it is "undirected", and "fuzzy". This adapted version is sometimes referred to as
  "normalized sum of distances" in the literature.
  Intuitively, it is the average euclidean distance (in pixels) each pixel in a prediction is
  from a pixel in the ground truth.

  This script can be run in isolation to compute the hausdorff distance between a prediction
  and ground truth volume (2D or 3D), given as arguments as follows:

  usage: python3 hausdorff_utils.py [threshold] ground_truth.TIFF prediction.TIFF

  IMPORTANT: This assumes that your image's pixel values are in the range [0, 255],
             if they are binary [0,1], then you must pass a threshold parameter --threshold=1

  3D hausdorff is not truly 3d, in this case we just compute the 2d hausdorff on each slice and average.
"""

import numpy as np
from scipy.spatial.distance import cdist, directed_hausdorff
import tifffile
import argparse

from tqdm import trange


def img_to_coordinate_list(img, threshold=50):
    """ pixels > threshold to [(x,y), (x,y)...]"""
    """ output format: [y, x] """
    return np.argwhere(img > threshold)


def hausdorff_vec(vector1, vector2, verbose=0):
    """
    :param vector1: [(x, y), (x, y)...] ; segmentation pixels
    :param vector2: [(x, y), (x, y)...] ; segmentation pixels
    :param verbose: 0, 1 or 2
    :return:
    """
    if verbose > 1:
        print(f'Annotation one has length {len(vector1)}')
        print(f'Annotation two has length {len(vector2)}')

    pwd = cdist(vector1, vector2)

    mins_XA = np.min(pwd, axis=0)
    mins_XB = np.min(pwd, axis=1)

    avg_XA = np.mean(mins_XA)
    avg_XB = np.mean(mins_XB)

    return (avg_XA + avg_XB) / 2


def hausdorff_2d(image1, image2, threshold=50, verbose=0):
    """
    :param image1: 2d image (segmentation map)
    :param image2: 2d image (segmentation map)
    :param threshold: threshold probabilistic images into hard/discreet
    segmentations. If the images are binary [0,1] then set this to 1.
    :param verbose: 0, 1 or 2
    :return: hausdorff distance between image one and image 2 interpreted
    as binary segmentations
    """

    coords1 = img_to_coordinate_list(image1, threshold)
    coords2 = img_to_coordinate_list(image2, threshold)
    return hausdorff_vec(coords1, coords2, verbose=verbose)


def hausdorff_3d(volume1, volume2, threshold=50, verbose=0):
    """
    :param volume1: 3d segmentation map image, depth in the first dimension
    :param volume2: 3d segmentation map image, depth in the first dimension
    :param threshold: hreshold probabilistic images into hard/discreet
    segmentations. If the images are binary [0,1] then set this to 1.
    :param verbose: 0, 1 or 2
    :return:
    """
    assert len(volume1) == len(volume2)

    num_slices = len(volume1)
    hausdorff_cumulative = 0
    n = 0  # number of slices with pixels > threhold

    for z in trange(num_slices):
        coords1 = img_to_coordinate_list(volume1[z], threshold)
        coords2 = img_to_coordinate_list(volume2[z], threshold)
        coords1_any = coords1.any()
        coords2_any = coords2.any()
        if coords1_any and coords2_any:
            distance = hausdorff_vec(coords1, coords2, verbose)
            hausdorff_cumulative += distance
            n += 1

        if verbose:
            print(f'\r{z}/{num_slices-1}', end='')  # progress

    hausdorff_avg = hausdorff_cumulative / n
    return hausdorff_avg


def hausdorff(image1, image2, threshold=50, verbose=0):
    """ combine 2d and 3d methods to handle arbitrary images """
    assert image1.shape == image2.shape
    if image1.ndim == 2:
        return hausdorff_2d(image1, image2, threshold, verbose)
    elif image1.ndim == 3:
        return hausdorff_3d(image1, image2, threshold, verbose)
    else:
        raise ValueError('Images must be either 2d or 3d.')


def max_hausdorff_dist(a, b):
    assert len(a) == len(b)

    num_slices = len(a)
    tota = 0
    n = 0

    for z in trange(num_slices):
        a_z = a[z]
        b_z = b[z]
        if a_z.any() and b_z.any():
            a1 = directed_hausdorff(a_z, b_z)[0]
            a2 = directed_hausdorff(b_z, a_z)[0]
            tota += (a1 + a2) / 2
            n += 1

    a = tota / n
    return a


def test():
    A = np.zeros((5, 5))
    B = np.zeros((5, 5))
    A[:, 3] = 100
    B[:, 0] = 100
    assert hausdorff_2d(A, B) == 3.0
    C = np.vstack([A,A,A])
    D = np.vstack([B,B,B])
    assert hausdorff_3d(C, D) == 3.0



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--threshold',
        type=int,
        help='threshold at which to consider a pixel part of the hard segmentation',
        default=50
    )
    parser.add_argument(
        '--verbose',
        type=int,
        help='print more info',
        default=0
    )
    parser.add_argument(
        'ground_truth',
        help='Ground truth image/volume, .tiff, 2d or 3d',
        type=str
    )
    parser.add_argument(
        'prediction',
        help='Prediction image/volume, .tiff, 2d or 3d',
        type=str
    )
    args = parser.parse_args()

    ground_truth_img = tifffile.imread(args.ground_truth)
    prediction_img = tifffile.imread(args.prediction)
    result = hausdorff(ground_truth_img, prediction_img, args.threshold, args.verbose)
    print(f'\n\nHausdorff distance: {result}')

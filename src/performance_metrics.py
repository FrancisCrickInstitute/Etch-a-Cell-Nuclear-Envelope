import numpy as np
import cv2
from tqdm import trange

from src.interiors_probability import get_closed_points, get_internal_area, get_distance, getContour_extrernal
from src.legacy.hausdorff_utils import hausdorff


def fmeasure(a, b):
    intersection = np.count_nonzero(a * b)
    return (2 * intersection) / (np.count_nonzero(a) + np.count_nonzero(b))


def mcc(y_true, y_pred):
    y_true_f = 1 - y_true
    y_pred_f = 1 - y_pred
    tp = np.count_nonzero(y_true * y_pred)
    fp = np.count_nonzero(y_true_f * y_pred)
    fn = np.count_nonzero(y_true * y_pred_f)
    tn = np.count_nonzero(y_true_f * y_pred_f)
    return (tp * tn - fp * fn) / np.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


def iou(a, b):
    intersection = np.count_nonzero(a * b)
    union = np.count_nonzero(np.logical_or(a > 0, b > 0) * 1)
    return intersection / union


def recall_precision(y_true, y_pred):
    intersection = np.count_nonzero(y_true * y_pred)
    recall = intersection / np.count_nonzero(y_true)
    precision = intersection / np.count_nonzero(y_pred)
    return recall, precision


def fmeasure_boundary(a, b):
    return measure_generic(a, b, fmeasure)


def fmeasure_area(a, b):
    return measure_generic(a, b, fmeasure_area_image)


def mcc_area(a, b):
    return measure_generic(a, b, mcc_area_image)


def measure_generic(a, b, function):
    assert len(a) == len(b)
    f = 0
    ftot = 0
    n = 0

    if a.ndim == 3:
        for z in trange(len(a)):
            a_z = a[z]
            b_z = b[z]
            if a_z.any() and b_z.any():
                ftot += function(a_z, b_z)
                n += 1
        if n:
            f = ftot / n
    else:
        f = function(a, b)
    return f


def fmeasure_area_image(a, b):
    a_area = boundary_to_area(a)
    b_area = boundary_to_area(b)
    return fmeasure(a_area, b_area)


def mcc_area_image(a, b):
    a_area = boundary_to_area(a)
    b_area = boundary_to_area(b)
    return mcc(a_area, b_area)


def boundary_to_area(image):
    area_image = np.zeros(image.shape, np.float32)
    if image.any():
        segments = getContour_extrernal(image)
        # flatten odd cv format
        segments2 = []
        for segment in segments:
            segments2.append(segment.reshape(len(segment), 2).tolist())
        closed_areas = get_closed_points(segments2)
        area_image = get_internal_area(closed_areas, image.shape[1], image.shape[0])
    return area_image.astype(np.float32)


def boundary_to_interior(boundary):
    # average over 3 axes
    interior = boundary_to_interior_3d(boundary)
    # rotate, boundary_to_interior_3d, rotate back, average

    return interior


def boundary_to_interior_3d(volume):
    interior = np.zeros(volume.shape, np.float32)
    for i in range(len(volume)):
        interior[i] = boundary_to_interior_2d(volume[i])
    return interior


def boundary_to_interior_2d(image):
    interior1 = boundary_to_interior_2d_1(image)
    interior2 = np.fliplr(boundary_to_interior_2d_1(np.fliplr(image)))
    interior = (interior1 + interior2) / 2
    interior[interior < 1] = 0
    return interior


def boundary_to_interior_2d_1(image):
    interior = np.zeros(image.shape, np.float32)
    for y in range(image.shape[0]):
        in_border = False
        accum = 0
        for x in range(image.shape[1]):
            if image[y][x] and not in_border:
                in_border = True
                accum += 1
            elif not image[y][x] and in_border:
                in_border = False
                accum += 1
            if (accum % 4) > 0:
                interior[y][x] = 1
    return interior


def get_connected_points(points, max_dist=1.5):
    point_lists = []
    points2 = []
    points = points.tolist()

    target_point = points[0]
    points2.append(target_point)
    points.remove(target_point)
    while points:
        mindist = -1
        close_point = []
        for find_point in points:
            dist = get_distance(find_point, target_point)
            if dist < mindist or mindist < 0:
                mindist = dist
                close_point = find_point
        if mindist >= 0:
            if mindist <= max_dist:
                points2.append(close_point)
                points.remove(close_point)
            else:
                if points2:
                    point_lists.append(points2)
                    points2 = []
            target_point = close_point

    if points2:
        point_lists.append(points2)

    return point_lists


def average_hausdorff_dist(a, b):
    return hausdorff(a, b, threshold=0)


def illustrate_area_images(output_path, y_true, y_pred):
    true_area = boundary_to_area(y_true)
    pred_area = boundary_to_area(y_pred)
    cv2.imwrite(output_path + "_true_area.tif", true_area)
    cv2.imwrite(output_path + "_pred_area.tif", pred_area)

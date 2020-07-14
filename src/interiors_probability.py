import os

import numpy as np
import math
import cv2
from PIL import Image, ImageDraw


def get_annotation_points(annotation_data, zoom_factor_x, zoom_factor_y):
    annotations = []
    for index, row in annotation_data.iterrows():
        row_annotations = []
        for annotation in eval(row.annotations):
            annotation_points = []
            for point in annotation:
                annotation_points.append([point[0] * zoom_factor_x, point[1] * zoom_factor_y])
            row_annotations.append(annotation_points)
        annotations.append(row_annotations)
    return annotations


def get_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def encode_point(segmenti, dir):
    s = str(segmenti)
    if dir == -1:
        s += "E"
    else:
        s += "S"
    return s


def get_closed_points(annotation):
    segment_distances = []
    final_segment_distances = []
    pointdir_done = []
    l = len(annotation)

    for f in range(l):
        for t in range(l):
            for fdir in range(-1, 1):   # last, first element
                for tdir in range(-1, 1):   # last, first element
                    if f is not t or fdir is not tdir:
                        segment_distances.append([f, fdir, t, tdir, get_distance(annotation[f][fdir], annotation[t][tdir])])

    segment_distances = sorted(segment_distances, key=lambda x: x[-1])

    for segment in segment_distances:
        f = segment[0]
        t = segment[2]
        fpdir = encode_point(segment[0], segment[1])
        tpdir = encode_point(segment[2], segment[3])
        if fpdir not in pointdir_done and tpdir not in pointdir_done:
            pointdir_done.append(fpdir)
            pointdir_done.append(tpdir)
            final_segment_distances.append(segment)

    #for segment_distance in final_segment_distances:
    #    print(encode_point(segment_distance[0], segment_distance[1]) +
    #          " <-> " + encode_point(segment_distance[2], segment_distance[3]) +
    #          " (" + str(segment_distance[-1]) + ")")

    dsegmenti_done = []
    segmenti = -1
    endsegmenti = -1
    pointslists = []
    points = []
    while len(dsegmenti_done) != l:
        if segmenti >= 0:
            i = 0
            for segment_distance in final_segment_distances:
                if i not in dsegmenti_done:
                    if segment_distance[0] == segmenti or segment_distance[2] == segmenti:
                        dsegmenti = i
                        dsegment = final_segment_distances[dsegmenti]
                        if segment_distance[0] == segmenti:
                            segmenti = segment_distance[2]
                        else:
                            segmenti = segment_distance[0]
                        break
                i += 1
        else:
            i = 0
            for segment_distance in final_segment_distances:
                if i not in dsegmenti_done:
                    dsegmenti = i
                    break
                i += 1
            dsegment = final_segment_distances[dsegmenti]
            segmenti = dsegment[0]
            endsegmenti = dsegment[2]

        dsegmenti_done.append(dsegmenti)

        segment = annotation[segmenti].copy()
        if dsegment[0] == segmenti:
            if dsegment[1] == -1:
                segment.reverse()
        else:
            if dsegment[3] == -1:
                segment.reverse()
        points.extend(segment)

        if segmenti == endsegmenti:
            # closed loop
            segmenti = -1
            endsegmenti = -1
            pointslists.append(points)
            points = []

    return pointslists


def to_cv_contours(closed_areas):
    # convert from list of lists to list of numpy arrays (each holding an array of points)
    cv_contour = []
    for closed_area in closed_areas:
        cv_contour.append(np.array(closed_area, dtype=np.int))
    return cv_contour


def get_internal_area(annotation, width, height):
    # fast contour to interior using opencv
    # output mat needs to be int type
    mat = np.zeros((height, width), np.uint8)
    cv_contours = to_cv_contours(annotation)
    cv2.fillPoly(mat, cv_contours, 1)
    return mat


def get_interiors_matrix(annotations, width, height):
    merged = np.zeros((height, width), np.float32)
    for annotation in annotations:
        closed_areas = get_closed_points(annotation)
        mat = get_internal_area(closed_areas, width, height)
        merged += mat
    return merged / len(annotations)


def threshold(mat):
    boolmat = (mat >= 0.5)
    return boolmat


def getContour(boolmat):
    mat = np.array(boolmat, np.uint8)
    contours, h = cv2.findContours(mat, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours


def getContour_extrernal(boolmat):
    mat = np.array(boolmat, np.uint8)
    contours, h = cv2.findContours(mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def draw_contours(contours, width, height, border_width):
    img = Image.new('F', (width, height), 0)
    draw = ImageDraw.Draw(img)
    for contour in contours:
        for j in range(1, len(contour)):
            draw.line([tuple(contour[j - 1][0]), tuple(contour[j][0])], fill=1, width=border_width)
    return np.asarray(img)


def draw_contours2(contours, width, height, border_width):
    image = np.zeros((height, width), np.float32)
    cv2.polylines(image, contours, False, 1, border_width - 1, lineType=cv2.LINE_AA)
    return image


def do_interiors_contours(annotations, width, height, border_width):
    mat = get_interiors_matrix(annotations, width, height)
    finalboolmat = threshold(mat)
    contours = getContour(finalboolmat)
    final = draw_contours2(contours, width, height, border_width)
    return final


def illustrate_draw_annotations(output_dir, annotations, width, height, border_width, is_closed=False):
    mat = np.zeros((height, width, 3), np.uint8)
    for annotation in annotations:
        if is_closed:
            cv_contours = to_cv_contours(get_closed_points(annotation))
        else:
            cv_contours = to_cv_contours(annotation)
        if len(annotations) == 1:
            color = (0xFF, 0xFF, 0xFF)
        else:
            color = tuple([int(x) for x in (0x0F + np.random.choice(range(0xF0), size=3))])
        cv2.polylines(mat, cv_contours, is_closed, color, border_width-1, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, "annotations.tif"), mat)


def illustrate_area_annotations(output_dir, annotations, width, height):
    mat = (get_interiors_matrix(annotations, width, height) * 0xFF).astype(np.uint8)
    mat = cv2.applyColorMap(mat, cv2.COLORMAP_HOT)
    cv2.imwrite(os.path.join(output_dir, "annotations_area.tif"), mat)

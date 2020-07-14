import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread, imsave
from skimage.transform import resize
from PIL import Image, ImageDraw
import matplotlib as mpl

from src.helpers import pad
from src.interiors_probability import get_annotation_points


mpl.rcParams['figure.dpi'] = 500

roi = "ROI_1656-6756-329"
Z_SLICE = 70


def load_annotations_as_images(csv_filename):
    df = pd.read_csv(csv_filename)
    annotations = get_annotation_points(df, 2, 2)

    annotation_images = []
    for annotation in annotations:
        annotation_img = Image.new('F', (2000, 2000), 0)
        draw = ImageDraw.Draw(annotation_img)

        for segment in annotation:
            prev = segment[0]
            for i in range(1, len(segment)):
                point = segment[i]
                draw.line([tuple(prev), tuple(point)], fill=1, width=3)
                prev = point

        annotation_images.append(np.asarray(annotation_img))
    return annotation_images


def load_annotations_as_lists(csv_filename):
    df = pd.read_csv(csv_filename)
    annotations = get_annotation_points(df, 2, 2)
    return annotations


if __name__ == '__main__':
    from matplotlib._color_data import CSS4_COLORS
    colors = list(CSS4_COLORS.keys())
    BORDER_WIDTH = 0.5
    WRITE_FOLDER = '../../projects/nuclear/resources/figures/'

    if not os.path.exists(WRITE_FOLDER):
        os.makedirs(WRITE_FOLDER)

    raw_stack_location = f"../../projects/nuclear/resources/images/backup-stacks/{roi}.tiff"
    csv_location = f"../../projects/nuclear/resources/csv/processed/{roi}_z{pad(str(Z_SLICE))}.csv"

    raw_stack = imread(raw_stack_location)

    # 1) raw image
    raw_image = raw_stack[Z_SLICE, :, :]

    plt.imshow(raw_image, cmap='gray')

    plt.axis('off')
    frame1_fn = os.path.join(WRITE_FOLDER, 'frame1.png')
    plt.savefig(frame1_fn, bbox_inches='tight')
    plt.cla()

    annotations_lists = load_annotations_as_lists(csv_location)
    # 2) all aggregations on top of raw image (with color)
    plt.imshow(raw_image, cmap='gray')
    for i, annotation_list in enumerate(annotations_lists):
        for segment in annotation_list:
            plt.plot(*zip(*segment), color=colors[i], linewidth=BORDER_WIDTH)

    plt.axis('off')
    frame2_fn = os.path.join(WRITE_FOLDER, 'frame2.png')
    plt.savefig(frame2_fn, bbox_inches='tight')
    plt.cla()

    # 3) zoomed in all aggregations (not greyscale heatmap as in existing image..)
    # annotations_images = load_annotations_as_images(csv_location)
    # heatmap = np.mean(annotations_images, axis=0)
    # #heatmap = np.logical_or.reduce(annotations_images, axis=0)
    # plt.imshow(heatmap, cmap='gray')
    # plt.show()
    #zoomed_image = raw_image[800:1000, 350:550]
    plt.xlim(350, 550)
    plt.ylim(800, 1000)
    plt.imshow(raw_image, cmap='gray')
    for i, annotation_list in enumerate(annotations_lists):
        for segment in annotation_list:
            plt.plot(*zip(*segment), color=colors[i], linewidth=3, alpha=0.5)

    plt.axis('off')
    frame3_fn = os.path.join(WRITE_FOLDER, 'frame3.png')
    plt.savefig(frame3_fn, bbox_inches='tight')
    plt.cla()

    # 4) I hate this on top of raw image
    ihatethis_idx = 14
    plt.imshow(raw_image, cmap='gray')
    for segment in annotations_lists[ihatethis_idx]:
        plt.plot(*zip(*segment), color=colors[ihatethis_idx], linewidth=BORDER_WIDTH)

    plt.axis('off')
    frame4_fn = os.path.join(WRITE_FOLDER, 'frame4.png')
    plt.savefig(frame4_fn, bbox_inches='tight')
    plt.cla()

    # 5) oversegmentation example
    oversegment_idx = 10
    plt.imshow(raw_image, cmap='gray')
    for segment in annotations_lists[oversegment_idx]:
        plt.plot(*zip(*segment), color=colors[oversegment_idx], linewidth=BORDER_WIDTH)

    plt.axis('off')
    frame5_fn = os.path.join(WRITE_FOLDER, 'frame5.png')
    plt.savefig(frame5_fn, bbox_inches='tight')
    plt.cla()

    # 6) undersegmentation example
    undersegment_idx = 15
    plt.imshow(raw_image, cmap='gray')
    for segment in annotations_lists[undersegment_idx]:
        plt.plot(*zip(*segment), color=colors[undersegment_idx], linewidth=BORDER_WIDTH)

    plt.axis('off')
    frame6_fn = os.path.join(WRITE_FOLDER, 'frame6.png')
    plt.savefig(frame6_fn, bbox_inches='tight')
    plt.cla()

    frame1 = imread(frame1_fn)
    frame2 = imread(frame2_fn)
    frame3 = imread(frame3_fn)
    frame4 = imread(frame4_fn)
    frame5 = imread(frame5_fn)
    frame6 = imread(frame6_fn)

    # rescale
    frame1_scaled = resize(frame1, (500, 500, 4))
    frame2_scaled = resize(frame2, (500, 500, 4))
    frame3_scaled = resize(frame3, (500, 500, 4))
    frame4_scaled = resize(frame4, (500, 500, 4))
    frame5_scaled = resize(frame5, (500, 500, 4))
    frame6_scaled = resize(frame6, (500, 500, 4))

    print(f"frame1_scaled shape: {frame1_scaled.shape}")
    print(f"frame2_scaled shape: {frame2_scaled.shape}")
    print(f"frame3_scaled shape: {frame3_scaled.shape}")
    print(f"frame4_scaled shape: {frame4_scaled.shape}")
    print(f"frame5_scaled shape: {frame5_scaled.shape}")
    print(f"frame6_scaled shape: {frame6_scaled.shape}")

    # combine
    combined = np.zeros((500*2, 500*3, 4))
    combined[0:500, 0:500] = frame1_scaled
    combined[0:500, 500:1000] = frame2_scaled
    combined[0:500, 1000:1500] = frame3_scaled
    combined[500:1000, 0:500] = frame4_scaled
    combined[500:1000, 500:1000] = frame5_scaled
    combined[500:1000, 1000:1500] = frame6_scaled

    imsave(os.path.join(WRITE_FOLDER, 'figure2.png'), combined)



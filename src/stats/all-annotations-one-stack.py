import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
import matplotlib as mpl
import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm

from src.interiors_probability import get_annotation_points, draw_contours, draw_contours2
from src.helpers import pad
from src.helpers import sizenm_to_dpum
from src.image_processing import save_image, to_binary

if __name__ == '__main__':
    mpl.rcParams['figure.dpi'] = 320

    #roi = "ROI_1656-6756-329"
    #roi = "ROI_3624-2712-201"
    roi = "ROI_1536-3456-213"
    raw_stack_location = f"../../projects/nuclear/resources/images/backup-stacks/{roi}.tiff"
    out_filename = f"../../projects/nuclear/resources/all_aggregations_{roi}.tiff"

    raw_stack = imread(raw_stack_location)
    out_stack = np.zeros(raw_stack.shape, dtype=np.float32)

    for idx in tqdm(range(raw_stack.shape[0])):
        csv_filename = f"../../projects/nuclear/resources/csv/processed/{roi}_z{pad(str(idx))}.csv"

        if os.path.exists(csv_filename):
            df = pd.read_csv(csv_filename)
            annotations = get_annotation_points(df, 2, 2)

            img = Image.new('F', (raw_stack.shape[2], raw_stack.shape[1]), 0)
            draw = ImageDraw.Draw(img)
            for annotation in annotations:
                cv_contours = []
                for segment in annotation:
                    prev = segment[0]
                    for i in range(1, len(segment)):
                        point = segment[i]
                        draw.line([tuple(prev), tuple(point)], fill=1, width=3)
                        prev = point

                    #cv_contours.append(np.array(segment, dtype=np.int))
                    #cv2.drawContours(out_stack[0], np.array(cv_contours), -1, 1, 7, lineType=cv2.LINE_AA)

            #img = img.resize((500, 500), resample=Image.BICUBIC)

            out_image = np.asarray(img)
            out_stack[idx] = out_image

    model_size_xy_nm = 10
    model_size_z_nm = 50

    resxy = sizenm_to_dpum(model_size_xy_nm)
    size_z_um = model_size_z_nm / 1000
    res_unit = "micron"

    out_stack = np.where(out_stack >= 0.5, 255, 0).astype(np.uint8)
    save_image(out_filename, out_stack, resx=resxy, resy=resxy, size_z=size_z_um,
               res_unit=res_unit, compress=True)


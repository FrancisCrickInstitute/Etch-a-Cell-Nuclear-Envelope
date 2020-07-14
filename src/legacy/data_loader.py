from skimage.io import imread
from os import path
import pandas as pd
import numpy as np


ANNOTATIONS_PATH = '23_may_2019_10nm.csv'
ROIs_PATH = 'ROI_IMAGE_VOLUMES'


class DataLoader:

  def __init__(self):
    self.df = pd.read_csv(ANNOTATIONS_PATH)

  def get_annotations(self, roi_id='ROI_1656-6756-329', slice_z=150):
    """
    :param df: dataframe (clean) containing all annotations
    :param roi_id: e.g. ROI_3000-3264-393
    :param slice_z: e.g. 150
    :return: (list) all rows from the dataframe of matching annotations
    default is for testing.
    """
    raw_slice_annotations = self.df.loc[(self.df['ROI'] == roi_id) & (self.df['slice z'] == slice_z)]
    grouped_annotations = raw_slice_annotations.groupby('classification id')
    as_list = [group for _, group in grouped_annotations]
    return as_list


def get_img_volume(roi_id='ROI_1656-6756-329'):
  """ simply load an image into a matrix """
  img_path = path.join(ROIs_PATH, f'{roi_id}.tiff')
  img_volume = imread(img_path)
  return img_volume


def annotation_to_vec(annotation):
  """ given a row from the dataframe, return a list of [(x1, y1), (x2, y2)...] coordinates """
  coordinate_list = np.vstack([np.array(eval(points)) for points in annotation['points']])
  return coordinate_list

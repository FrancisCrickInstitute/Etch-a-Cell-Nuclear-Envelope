import numpy as np


def iou(ground_truth, prediction):
  """
  intersection over union
  aka jaccard index, aka dice score
  """
  intersection = ground_truth * ground_truth
  union = np.sum(ground_truth) + np.sum(prediction)
  score = intersection.sum() / union
  return score

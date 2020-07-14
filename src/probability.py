
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp
from skimage import filters


def P(E, T=1.0):
    """
    Calculate the probability of being in the state denoted by the variable E.

    Z can be seen as a two state partition function with an unoccupied state of zero energy
    and an occupied state with E energy. Lower energy increases the probability of being in
    the occupied state. In any case P(E) reduces to a sigmoid function.

    Here the occupied state would correspond to a pixel which is part of the segmentation
    and the unoccupied state to a pixel which is not part of the segmentation.

    T adds noise which translates into making the function less steep.
    """
    Z = 1 + exp(-E / T)
    return exp(-E / T) / Z


def probabilistic_aggregate_1x1(matrices, threshold=0.05, target_ratio=0.5, T=0.125):
    """
    For each pixel in the input matrix shape, calculate the number of matrices which believe that pixel is part of the
    segmentation. Then divide that number by the number of matrices, thereby getting an occupancy ratio. This then
    forms the variable to feed in to a suitably parameterized sigmoid function.
    """
    groups, rows, cols = matrices.shape

    energy_matrix = target_ratio - np.sum(matrices, axis=0) / groups
    prob = P(energy_matrix, T=T)

    if threshold is None or threshold == 'otsu':
        threshold = filters.threshold_otsu(prob)
    prob[prob < threshold] = 0

    return prob


def plot_probability_matrix(probability_matrix, title='', image=None, alpha=0.3):
    if image is not None:
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.gca().set_title(title)
    filtered_probability_matrix = np.ma.masked_where(probability_matrix == 0, probability_matrix)
    plt.imshow(filtered_probability_matrix * 255, cmap='cool', vmin=0, vmax=255, alpha=alpha)
    plt.show()


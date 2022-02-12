import numpy as np


def pixel_accuracy(target, prediction):
    return np.sum(target == prediction) / target.size


def intersection_over_union(target, prediction):
    intersection = np.sum(target & prediction)
    union = np.sum(target | prediction)
    return intersection / union

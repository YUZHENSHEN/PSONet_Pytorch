# coding=utf-8

import sys
import numpy as np

# sample selection
def sampleselect(WTSCS, pred, mask):
    """
    Select samples in each progression.
    Args:
        WTSCS: The “weak-to-strong” change signal image generated in last progression
        pred: Prediction results from last progression
        mask: Index of test samples in last progression
    Returns:
        index0: Unchanged samples selected in current progression
        index1: Changed samples selected in current progression
        index2: Unknown samples selected in current progression
        WTSCS: The updated “weak-to-strong” change signal image
    """
    if WTSCS.shape != mask.shape:
        print("note:-----the shape of pre_result and post_result is different-----")
        sys.exit()
    else:
        nr, nc = WTSCS.shape
        WTSCS = np.array(WTSCS * mask + pred, dtype=int)
        unique_values = np.unique(WTSCS)
        print("note:-----the unique values in this progression:", unique_values)
        unique_values_without0 = unique_values[unique_values != 0]
        print("note:-----the unique values without 0 in this progression:", unique_values_without0)
        if len(unique_values_without0) == 1 and unique_values_without0[0] > 5:
            index0 = np.zeros((nr, nc), dtype=int)
            max_value = unique_values_without0[0]
            index1 = np.reshape(WTSCS == max_value, newshape=(nr * nc,))
            index2 = np.zeros((nr, nc), dtype=int)
        elif len(unique_values_without0) == 1 and unique_values_without0[0] <= 5:
            min_value = unique_values_without0[0]
            index0 = np.reshape(WTSCS == min_value, newshape=(nr * nc,))
            index1 = np.zeros((nr, nc), dtype=int)
            index2 = np.zeros((nr, nc), dtype=int)
        else:
            min_value = np.min(unique_values_without0)
            max_value = np.max(unique_values_without0)
            index0 = np.reshape(WTSCS == min_value, newshape=(nr * nc,))  # unchanged
            index1 = np.reshape(WTSCS == max_value, newshape=(nr * nc,))  # change
            index2 = np.reshape((WTSCS > min_value) & (WTSCS < max_value), newshape=(nr * nc,))
        return index0, index1, index2, WTSCS
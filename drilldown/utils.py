import collections.abc
import numpy as np
import pandas as pd


def convert_to_numpy_array(arr, collapse_dim=True):
    if isinstance(arr, pd.core.series.Series):
        arr = arr.values
    elif isinstance(arr, pd.core.frame.DataFrame):
        n_cols = arr.shape[1]
        if n_cols == 1:
            if collapse_dim == True:
                arr = arr.values[:, 0]
            else:
                arr = arr.values
        else:
            arr = arr.values
    elif isinstance(arr, collections.abc.Sequence):
        arr = np.array(arr)
    elif isinstance(arr, np.ndarray):
        pass
    else:
        raise TypeError("Data must be a sequence or Pandas object.")

    return arr

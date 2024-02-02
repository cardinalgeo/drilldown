import collections.abc
import numpy as np
import pandas as pd
import distinctipy
from matplotlib.colors import ListedColormap


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


def convert_array_type(arr, return_type=False):
    try:
        arr = arr.astype("float")
        _type = "float"
    except ValueError:
        arr = arr.astype("str")
        _type = "str"

    if return_type == True:
        return arr, _type
    else:
        return arr


def encode_categorical_data(data):
    data = convert_to_numpy_array(data)

    data_encoded, categories = pd.factorize(data)
    codes = np.arange(len(categories))

    # convert codes to float
    codes = np.array(codes, dtype="float")
    data_encoded = np.array(data_encoded, dtype="float")

    # # center numerical representation of categorical data, while maintaining range, to address pyvista's color mapping querks
    # codes[1:-1] += 0.5
    # data_encoded[
    #     (data_encoded != data_encoded.min()) & (data_encoded != data_encoded.max())
    # ] += 0.5
    code_to_cat_map = {code: cat for code, cat in zip(codes, categories)}

    return code_to_cat_map, data_encoded


def get_cycled_colors(n):
    from itertools import cycle
    import matplotlib as mpl

    cycler = mpl.rcParams["axes.prop_cycle"]
    colors = cycle(cycler)
    colors = [next(colors)["color"] for i in range(n)]
    colors = [mpl.colors.to_rgb(color) for color in colors]

    return colors


def make_color_map_fractional(map):
    for name in map.keys():
        if max(map[name]) > 1:
            map[name] = tuple([val / 255 for val in map[name]])
        elif max(map[name]) >= 0:
            # raise ValueError("Color is already fractional.")
            pass
        else:
            raise ValueError("Color values cannot be negative.")

    return map


def make_categorical_cmap(categories, cycle=True, rng=999, pastel_factor=0.2):
    n_colors = len(categories)
    if cycle == True:
        colors = get_cycled_colors(n_colors)

    elif cycle == False:
        colors = distinctipy.get_colors(
            n_colors,
            pastel_factor=pastel_factor,
            rng=rng,
        )

    # create categorical color map
    cat_to_color_map = {cat: color for cat, color in zip(categories, colors)}

    # create matplotlib categorical color map
    matplotlib_formatted_color_maps = ListedColormap(colors)

    return cat_to_color_map, matplotlib_formatted_color_maps


def is_jupyter():
    from IPython import get_ipython

    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False

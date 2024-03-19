import pandas as pd


# points by points
def filter_points_by_points(depths_1, depths_2, tolerance=0.001):
    """Filter points at one set of depths by points at another set of depths (i.e., identify the intersection of the two datasets by depth).

    Parameters
    ----------
    depths_1 : pandas.DataFrame
        A DataFrame containing a column of depths to which the filter will be applied

    depths_2 : list or pandas.Series
        A list or Series containing depths. Used to apply a filter to the points corresponding to `depths_1`.

    tolerance : float, optional
        The tolerance for identifying equivalent depths, by default 0.001.

    Returns
    -------
    list
        A list of boolean values indicating whether each point in `depths_1` is within tolerance of any point in `depths_2`.

    """
    overlap_fun = lambda row: any(
        ((depth - row["depth"] > tolerance) and (row["depth"] - depth > tolerance))
        for depth in depths_2
    )

    overlap_filter = depths_1.apply(overlap_fun, axis=1).tolist()

    return overlap_filter


def select_points_by_points(depths_1, depths_2, tolerance=0.001):
    """Select points at one set of depths by points at another set of depths (i.e., identify the intersection of the two datasets by depth).

    Parameters
    ----------
    depths_1 : pandas.DataFrame
        A DataFrame containing a column of depths on which the selection will be performed

    depths_2 : list or pandas.Series
        A list or Series containing depths. Used to perform a selection on the points corresponding to `depths_1`.

    tolerance : float, optional
        The tolerance for identifying equivalent depths, by default 0.001.

    Returns
    -------
    list
        A list of indices of points in `depths_1` that are within tolerance of any point in `depths_2`.

    """
    overlap_filter = filter_points_by_points(depths_1, depths_2, tolerance)
    overlap_selection = depths_1[overlap_filter].index.tolist()

    return overlap_selection


def filter_hole_points_by_points(hole_id, depths_1, depths_2, tolerance=0.001):
    """For each hole, filter points at one set of depths by points at another set of depths.

    Parameters
    ----------
    hole_id : array-like
        An array-like object containing hole IDs corresponding to the depths in `depths_1`. Must be the same length as `depths_1`.

    depths_1 : pandas.DataFrame
        A DataFrame containing a column of depths corresponding to points to which the filter will be applied

    depths_2 : dict
        A dictionary for which each key is a unique hole ID and the corresponding value is a pandas.Series containing depths for that hole. Used to apply a filter to the points corresponding to `depths_1`.

    tolerance : float, optional
        The tolerance for identifying equivalent depths, by default 0.001.

    Returns
    -------
    list
        A list of boolean values indicating whether each point in `depths_1` is within tolerance of any point in `depths_2` for each hole.

    """
    overlap_filter = pd.Series(index=hole_id.index, data=False)
    hole_ids = [hole for hole in list(set(hole_id)) if hole in depths_2.keys()]
    for hole in hole_ids:
        hole_filter = hole_id == hole
        hole_depths_1 = depths_1[hole_filter]
        hole_depths_2 = depths_2[hole]
        overlap_filter[hole_filter] = filter_points_by_points(
            hole_depths_1, hole_depths_2, tolerance
        )

    return overlap_filter.tolist()


def select_hole_points_by_points(hole_id, depths_1, depths_2, tolerance=0.001):
    """For each hole, select points at one set of depths by points at another set of depths.

    Parameters
    ----------
    hole_id : array-like
        An array-like object containing hole IDs corresponding to the depths in `depths_1`. Must be the same length as `depths_1`.

    depths_1 : pandas.DataFrame
        A DataFrame containing a column of depths corresponding to points on which the selection will be performed

    depths_2 : dict
        A dictionary for which each key is a unique hole ID and the corresponding value is a pandas.Series containing depths for that hole. Used to perform a selection on the points corresponding to `depths_1`.

    tolerance : float, optional
        The tolerance for identifying equivalent depths, by default 0.001.

    Returns
    -------
    list
        A list of indices of points in `depths_1` that are within tolerance of any point in `depths_2` for each hole.

    """
    selected_ids = []
    hole_ids = [hole for hole in list(set(hole_id)) if hole in depths_2.keys()]
    for hole in hole_ids:
        hole_filter = hole_id == hole
        hole_depths_1 = depths_1[hole_filter]
        hole_depths_2 = depths_2[hole]
        selected_ids += select_points_by_points(hole_depths_1, hole_depths_2, tolerance)

    return selected_ids


# points by intervals
def filter_points_by_intervals(depths, ranges, tolerance=0.001):
    """Filter points at one set of depths by intervals, each of which is constrained by a "from" depth and "to" depth

    Identifies all points contained within any of the intervals defined by the "from" and "to" depths.

    Parameters
    ----------
    depths : pandas.DataFrame
        A DataFrame containing a column of depths corresponding to points to which the filter will be applied

    ranges : pandas.DataFrame
        A DataFrame containing columns of "from" and "to" depths corresponding to intervals. Used to apply a filter to the points corresponding to `depths`.

    tolerance : float, optional
        The tolerance for identifying equivalent depths, by default 0.001.

    Returns
    -------
    list
        A list of boolean values indicating whether each point in `depths` is within any of the intervals defined by `ranges`.

    """
    overlap_fun = lambda row: any(
        (
            (row["depth"] - from_depth > tolerance)
            and (to_depth - row["depth"] > tolerance)
        )
        for from_depth, to_depth in ranges
    )

    overlap_filter = depths.apply(overlap_fun, axis=1).tolist()

    return overlap_filter


def select_points_by_intervals(depths, ranges, tolerance=0.001):
    """Select points at one set of depths by intervals, each of which is constrained by a "from" depth and "to" depth

    Identifies all points contained within any of the intervals defined by the "from" and "to" depths.

    Parameters
    ----------
    depths : pandas.DataFrame
        A DataFrame containing a column of depths corresponding to points on which the selection will be performed

    ranges : pandas.DataFrame
        A DataFrame containing columns of "from" and "to" depths corresponding to intervals. Used to perform a selection on the points corresponding to `depths`.

    tolerance : float, optional
        The tolerance for identifying equivalent depths, by default 0.001.

    Returns
    -------
    list
        A list of indices of points in `depths` that are within any of the intervals defined by `ranges`.

    """
    overlap_filter = filter_points_by_intervals(depths, ranges, tolerance)
    overlap_selection = depths[overlap_filter].index.tolist()

    return overlap_selection


def filter_hole_points_by_intervals(hole_id, depths, ranges, tolerance=0.001):
    """For each hole, filter points at one set of depths by intervals, each of which is constrained by a "from" depth and "to" depth

    Identifies all points contained within any of the intervals defined by the "from" and "to" depths, for each hole.

    Parameters
    ----------
    hole_id : array-like
        An array-like object containing hole IDs corresponding to the depths in `depths`. Must be the same length as `depths`.

    depths : pandas.DataFrame
        A DataFrame containing a column of depths corresponding to points to which the filter will be applied

    ranges : dict
        A dictionary for which each key is a unique hole ID and the corresponding value is a pandas.DataFrame containing columns of "from" and "to" depths corresponding to intervals. Used to apply a filter to the points corresponding to depths.

    tolerance : float, optional
        The tolerance for identifying equivalent depths, by default 0.001.

    Returns
    -------
    list
        A list of boolean values indicating whether each point in depths is within any of the intervals defined by ranges, for each hole.

    """
    overlap_filter = pd.Series(index=hole_id.index, data=False)
    hole_ids = [hole for hole in list(set(hole_id)) if hole in ranges.keys()]
    for hole in hole_ids:
        hole_filter = hole_id == hole
        hole_depths = depths[hole_filter]
        hole_ranges = ranges[hole]
        overlap_filter[hole_filter] = filter_points_by_intervals(
            hole_depths, hole_ranges, tolerance
        )

    return overlap_filter.tolist()


def select_hole_points_by_intervals(hole_id, depths, ranges, tolerance=0.001):
    """For each hole, select points at one set of depths by intervals, each of which is constrained by a "from" depth and "to" depth

    Identifies all points contained within any of the intervals defined by the "from" and "to" depths, for each hole.

    Parameters
    ----------
    hole_id : array-like
        An array-like object containing hole IDs corresponding to the depths in `depths`. Must be the same length as `depths`.

    depths : pandas.DataFrame
        A DataFrame containing a column of depths corresponding to points on which the selection will be performed

    ranges : dict
        A dictionary for which each key is a unique hole ID and the corresponding value is a pandas.DataFrame containing columns of "from" and "to" depths corresponding to intervals. Used to perform a selection on the points corresponding to `depths`.

    tolerance : float, optional
        The tolerance for identifying equivalent depths, by default 0.001.

    Returns
    -------
    list
        A list of indices of points in `depths` that are within any of the intervals defined by `ranges`, for each hole.

    """
    selected_ids = []
    hole_ids = [hole for hole in list(set(hole_id)) if hole in ranges.keys()]
    for hole in hole_ids:
        hole_filter = hole_id == hole
        hole_depths = depths[hole_filter]
        hole_ranges = ranges[hole]
        selected_ids += select_points_by_intervals(hole_depths, hole_ranges, tolerance)

    return selected_ids


# intervals by intervals
def filter_intervals_by_intervals(from_to, ranges, overlap="partial", tolerance=0.001):
    """Filter intervals by intervals

    The filter can be applied considering "partial" overlap, for which the intervals to which the filter is applied need only be partially overlapped by the "filtering" intervals, or "complete" overlap, for which the intervals to which the filter is applied must be completely overlapped by the "filtering" intervals.

    Parameters
    ----------
    from_to : pandas.DataFrame
        A DataFrame containing columns of "from" and "to" depths corresponding to intervals to which the filter will be applied

    ranges : pandas.DataFrame
        A DataFrame containing columns of "from" and "to" depths corresponding to intervals. Used to apply a filter to the intervals corresponding to `from_to`.

    overlap : str, optional
        The type of overlap to consider. Must be one of "partial" or "complete", by default "partial"

    tolerance : float, optional
        The tolerance for identifying equivalent depths, by default 0.001.

    Returns
    -------
    list
        A list of boolean values indicating whether each interval in `from_to` that are overlapped by any of the intervals defined by `ranges`.

    """
    partial_fun = lambda row: any(
        (
            not (
                (row["from"] - to_depth > -tolerance)
                or (from_depth - row["to"] > -tolerance)
            )
        )
        for from_depth, to_depth in ranges
    )
    complete_fun = lambda row: any(
        (
            (
                not (
                    (row["from"] - to_depth > -tolerance)
                    or (from_depth - row["to"] > -tolerance)
                )
                and (row["from"] - from_depth > -tolerance)
                and (to_depth - row["to"] > -tolerance)
            )
        )
        for from_depth, to_depth in ranges
    )

    if overlap == "complete":
        overlap_fun = complete_fun
    elif overlap == "partial":
        overlap_fun = partial_fun

    overlap_filter = from_to.apply(overlap_fun, axis=1).tolist()

    return overlap_filter


def select_intervals_by_intervals(from_to, ranges, overlap="partial", tolerance=0.001):
    """Select intervals by intervals

    The selection can be performed considering "partial" overlap, for which the intervals on which the selection is performed need only be partially overlapped by the "selecting" intervals, or "complete" overlap, for which the intervals on which the selection is performed must be completely overlapped by the "selecting" intervals.

    Parameters
    ----------
    from_to : pandas.DataFrame
        A DataFrame containing columns of "from" and "to" depths corresponding to intervals on which the selection will be performed

    ranges : pandas.DataFrame
        A DataFrame containing columns of "from" and "to" depths corresponding to intervals. Used to perform a selection on the intervals corresponding to `from_to`.

    overlap : str, optional
        The type of overlap to consider. Must be one of "partial" or "complete", by default "partial"

    tolerance : float, optional
        The tolerance for identifying equivalent depths, by default 0.001.

    Returns
    -------
    list
        A list of indices of intervals in `from_to` that are overlapped by any of the intervals defined by `ranges`.

    """
    overlap_filter = filter_intervals_by_intervals(from_to, ranges, overlap, tolerance)
    overlap_selection = from_to[overlap_filter].index.tolist()

    return overlap_selection


def filter_hole_intervals_by_intervals(
    hole_id, from_to, ranges, overlap="partial", tolerance=0.001
):
    """For each hole, filter intervals by intervals

    The filter can be applied considering "partial" overlap, for which the intervals to which the filter is applied need only be partially overlapped by the "filtering" intervals, or "complete" overlap, for which the intervals to which the filter is applied must be completely overlapped by the "filtering" intervals.

    Parameters
    ----------
    hole_id : array-like
        An array-like object containing hole IDs corresponding to the intervals in `from_to`. Must be the same length as `from_to`.

    from_to : pandas.DataFrame
        A DataFrame containing columns of "from" and "to" depths corresponding to intervals to which the filter will be applied

    ranges : dict
        A dictionary for which each key is a unique hole ID and the corresponding value is a pandas.DataFrame containing columns of "from" and "to" depths corresponding to intervals. Used to apply a filter to the intervals corresponding to `from_to`.

    overlap : str, optional
        The type of overlap to consider. Must be one of "partial" or "complete", by default "partial"

    tolerance : float, optional

    Returns
    -------
    list
        A list of boolean values indicating whether each interval in `from_to` is overlapped by any of the intervals defined by `ranges`, for each hole.

    """

    overlap_filter = pd.Series(index=hole_id.index, data=False)
    hole_ids = [hole for hole in list(set(hole_id)) if hole in ranges.keys()]
    for hole in hole_ids:
        hole_filter = hole_id == hole
        hole_from_to = from_to[hole_filter]
        hole_ranges = ranges[hole]
        overlap_filter[hole_filter] = filter_intervals_by_intervals(
            hole_from_to, hole_ranges, overlap, tolerance
        )

    return overlap_filter.tolist()


def select_hole_intervals_by_intervals(
    hole_id, from_to, ranges, overlap="partial", tolerance=0.001
):
    """For each hole, select intervals by intervals

    The selection can be performed considering "partial" overlap, for which the intervals on which the selection is performed need only be partially overlapped by the "selecting" intervals, or "complete" overlap, for which the intervals on which the selection is performed must be completely overlapped by the "selecting" intervals.

    Parameters
    ----------
    hole_id : array-like
        An array-like object containing hole IDs corresponding to the intervals in `from_to`. Must be the same length as `from_to`.

    from_to : pandas.DataFrame
        A DataFrame containing columns of "from" and "to" depths corresponding to intervals on which the selection will be performed

    ranges : dict
        A dictionary for which each key is a unique hole ID and the corresponding value is a pandas.DataFrame containing columns of "from" and "to" depths corresponding to intervals. Used to perform a selection on the intervals corresponding to `from_to`.

    overlap : str, optional
        The type of overlap to consider. Must be one of "partial" or "complete", by default "partial"

    tolerance : float, optional
        The tolerance for identifying equivalent depths, by default 0.001.

    Returns
    -------
    list
        A list of indices of intervals in `from_to` that are overlapped by any of the intervals defined by `ranges`, for each hole.

    """
    selected_ids = []
    hole_ids = [hole for hole in list(set(hole_id)) if hole in ranges.keys()]
    for hole in hole_ids:
        hole_filter = hole_id == hole
        hole_from_to = from_to[hole_filter]
        hole_ranges = ranges[hole]
        selected_ids += select_intervals_by_intervals(
            hole_from_to, hole_ranges, overlap, tolerance
        )

    return selected_ids


# intervals by points
def filter_intervals_by_points(from_to, depths, tolerance=0.001):
    """Filter intervals by points

    Identifies all intervals that contain any of the points defined by `depths`

    Parameters
    ----------
    from_to : pandas.DataFrame
        A DataFrame containing columns of "from" and "to" depths corresponding to intervals to which the filter will be applied

    depths : list or pandas.Series
        A list or Series containing depths. Used to apply a filter to the intervals corresponding to `from_to`.

    tolerance : float, optional
        The tolerance for identifying equivalent depths, by default 0.001.

    Returns
    -------
    list
        A list of boolean values indicating whether each interval in `from_to` contains any of the points defined by `depths`.

    """
    overlap_fun = lambda row: any(
        ((depth - row["from"] > tolerance) and (row["to"] - depth > tolerance))
        for depth in depths
    )

    overlap_filter = from_to.apply(overlap_fun, axis=1).tolist()

    return overlap_filter


def select_intervals_by_points(from_to, depths, tolerance=0.001):
    """Select intervals by points

    Identifies all intervals that contain any of the points defined by `depths`

    Parameters
    ----------
    from_to : pandas.DataFrame
        A DataFrame containing columns of "from" and "to" depths corresponding to intervals on which the selection will be performed

    depths : list or pandas.Series
        A list or Series containing depths. Used to perform a selection on the intervals corresponding to `from_to`.

    tolerance : float, optional
        The tolerance for identifying equivalent depths, by default 0.001.

    Returns
    -------
    list
        A list of indices of intervals in `from_to` that contain any of the points defined by `depths`.

    """
    overlap_filter = filter_intervals_by_points(from_to, depths, tolerance)
    overlap_selection = from_to[overlap_filter].index.tolist()

    return overlap_selection


def filter_hole_intervals_by_points(hole_id, from_to, depths, tolerance=0.001):
    """For each hole, filter intervals by points

    Identifies all intervals that contain any of the points defined by `depths`, for each hole.

    Parameters
    ----------
    hole_id : array-like
        An array-like object containing hole IDs corresponding to the intervals in `from_to`. Must be the same length as `from_to`.

    from_to : pandas.DataFrame
        A DataFrame containing columns of "from" and "to" depths corresponding to intervals to which the filter will be applied

    depths : dict
        A dictionary for which each key is a unique hole ID and the corresponding value is a pandas.Series containing depths for that hole. Used to apply a filter to the intervals corresponding to `from_to`.

    tolerance : float, optional
        The tolerance for identifying equivalent depths, by default 0.001.

    Returns
    -------
    list
        A list of boolean values indicating whether each interval in `from_to` contains any of the points defined by `depths`, for each hole.

    """
    overlap_filter = pd.Series(index=hole_id.index, data=False)
    hole_ids = [hole for hole in list(set(hole_id)) if hole in depths.keys()]
    for hole in hole_ids:
        hole_filter = hole_id == hole
        hole_from_to = from_to[hole_filter]
        hole_depths = depths[hole]
        overlap_filter[hole_filter] = filter_intervals_by_points(
            hole_from_to, hole_depths, tolerance
        )

    return overlap_filter.tolist()


def select_hole_intervals_by_points(hole_id, from_to, depths, tolerance=0.001):
    """For each hole, select intervals by points

    Identifies all intervals that contain any of the points defined by `depths`, for each hole.

    Parameters
    ----------
    hole_id : array-like
        An array-like object containing hole IDs corresponding to the intervals in `from_to`. Must be the same length as `from_to`.

    from_to : pandas.DataFrame
        A DataFrame containing columns of "from" and "to" depths corresponding to intervals on which the selection will be performed

    depths : dict
        A dictionary for which each key is a unique hole ID and the corresponding value is a pandas.Series containing depths for that hole. Used to perform a selection on the intervals corresponding to `from_to`.

    tolerance : float, optional
        The tolerance for identifying equivalent depths, by default 0.001.

    Returns
    -------
    list
        A list of indices of intervals in `from_to` that contain any of the points defined by `depths`, for each hole.

    """
    selected_ids = []
    hole_ids = [hole for hole in list(set(hole_id)) if hole in depths.keys()]
    for hole in hole_ids:
        hole_filter = hole_id == hole
        hole_from_to = from_to[hole_filter]
        hole_depths = depths[hole]
        selected_ids += select_intervals_by_points(hole_from_to, hole_depths, tolerance)

    return selected_ids


# Point Mixin
class PointInterLayerMixin:
    """A mixin class to manager interactions between point layers and other layers"""

    def filter_by_selection(self, layer, tolerance=0.001):
        """Filter points by a selection performed on another layer

        Parameters
        ----------
        layer : drilldown.layer.IntervalDataLayer or drilldown.layer.PointDataLayer
            The layer whose selection will be used to filter the points

        tolerance : float, optional
            The tolerance for identifying equivalent depths, by default 0.001.

        """
        from .layer import IntervalDataLayer, PointDataLayer

        depths = self.data[["depth"]]
        hole_id = self.data["hole ID"]

        selected_data = layer.selected_data

        if isinstance(layer, IntervalDataLayer):
            ranges = {}
            for hole in layer.selected_hole_ids:
                ranges[hole] = selected_data[selected_data["hole ID"] == hole][
                    ["from", "to"]
                ].values

            overlap_filter = filter_hole_points_by_intervals(
                hole_id, depths, ranges, tolerance
            )

        elif isinstance(layer, PointDataLayer):
            depths_2 = {}
            for hole in layer.selected_hole_ids:
                depths_2[hole] = selected_data[selected_data["hole ID"] == hole][
                    "depth"
                ]

            overlap_filter = filter_hole_points_by_points(
                hole_id, depths, depths_2, tolerance
            )

        self.boolean_filter = overlap_filter

    def select_by_selection(self, layer, tolerance=0.001):
        """Select points by a selection performed on another layer

        Parameters
        ----------
        layer : drilldown.layer.IntervalDataLayer or drilldown.layer.PointDataLayer
            The layer whose selection will be used to select the points

        tolerance : float, optional
            The tolerance for identifying equivalent depths, by default 0.001.

        """
        from .layer import IntervalDataLayer, PointDataLayer

        depths = self.data[["depth"]]
        hole_id = self.data["hole ID"]

        selected_data = layer.selected_data

        if isinstance(layer, IntervalDataLayer):
            ranges = {}
            for hole in layer.selected_hole_ids:
                ranges[hole] = selected_data[selected_data["hole ID"] == hole][
                    ["from", "to"]
                ].values

            selected_ids = select_hole_points_by_intervals(
                hole_id, depths, ranges, tolerance
            )

        elif isinstance(layer, PointDataLayer):
            depths_2 = {}
            for hole in layer.selected_hole_ids:
                depths_2[hole] = selected_data[selected_data["hole ID"] == hole][
                    "depth"
                ]

            selected_ids = select_hole_points_by_points(
                hole_id, depths, depths_2, tolerance
            )

        self.selected_ids = selected_ids

    def filter_by_filter(self, layer, tolerance=0.001):
        """Filter points by a filter applied to another layer

        Parameters
        ----------
        layer : drilldown.layer.IntervalDataLayer or drilldown.layer.PointDataLayer
            The layer whose filter will be used to filter the points

        tolerance : float, optional
            The tolerance for identifying equivalent depths, by default 0.001.

        """
        from .layer import IntervalDataLayer, PointDataLayer

        depths = self.data[["depth"]]
        hole_id = self.data["hole ID"]

        filtered_data = layer.filtered_data

        if isinstance(layer, IntervalDataLayer):
            ranges = {}
            for hole in layer.filtered_hole_ids:
                ranges[hole] = filtered_data[filtered_data["hole ID"] == hole][
                    ["from", "to"]
                ].values

            overlap_filter = filter_hole_points_by_intervals(
                hole_id, depths, ranges, tolerance
            )

        elif isinstance(layer, PointDataLayer):
            depths_2 = {}
            for hole in layer.filtered_hole_ids:
                depths_2[hole] = filtered_data[filtered_data["hole ID"] == hole][
                    "depth"
                ]

            overlap_filter = filter_hole_points_by_points(
                hole_id, depths, depths_2, tolerance
            )

        self.boolean_filter = overlap_filter

    def select_by_filter(self, layer, tolerance=0.001):
        """Select points by a filter applied to another layer

        Parameters
        ----------
        layer : drilldown.layer.IntervalDataLayer or drilldown.layer.PointDataLayer
            The layer whose filter will be used to select the points

        tolerance : float, optional
            The tolerance for identifying equivalent depths, by default 0.001.

        """
        from .layer import IntervalDataLayer, PointDataLayer

        depths = self.data[["depth"]]
        hole_id = self.data["hole ID"]

        filtered_data = layer.filtered_data

        if isinstance(layer, IntervalDataLayer):
            ranges = {}
            for hole in layer.filtered_hole_ids:
                ranges[hole] = filtered_data[filtered_data["hole ID"] == hole][
                    ["from", "to"]
                ].values

            selected_ids = select_hole_points_by_intervals(
                hole_id, depths, ranges, tolerance
            )

        elif isinstance(layer, PointDataLayer):
            depths_2 = {}
            for hole in layer.filtered_hole_ids:
                depths_2[hole] = filtered_data[filtered_data["hole ID"] == hole][
                    "depth"
                ]

            selected_ids = select_hole_points_by_points(
                hole_id, depths, depths_2, tolerance
            )

        self.selected_ids = selected_ids

    def filter_by(self, layer, tolerance=0.001):
        """Filter points by another layer

        Parameters
        ----------
        layer : drilldown.layer.IntervalDataLayer or drilldown.layer.PointDataLayer
            The layer used to filter the points

        tolerance : float, optional
            The tolerance for identifying equivalent depths, by default 0.001.

        """
        from .layer import IntervalDataLayer, PointDataLayer

        depths = self.data[["depth"]]
        hole_id = self.data["hole ID"]

        data = layer.data

        if isinstance(layer, IntervalDataLayer):
            ranges = {}
            for hole in layer.filtered_hole_ids:
                ranges[hole] = data[data["hole ID"] == hole][["from", "to"]].values

            overlap_filter = filter_hole_points_by_intervals(
                hole_id, depths, ranges, tolerance
            )

        elif isinstance(layer, PointDataLayer):
            depths_2 = {}
            for hole in layer.filtered_hole_ids:
                depths_2[hole] = data[data["hole ID"] == hole]["depth"]

            overlap_filter = filter_hole_points_by_points(
                hole_id, depths, depths_2, tolerance
            )

        self.boolean_filter = overlap_filter

    def select_by(self, layer, tolerance=0.001):
        """Select points by another layer

        Parameters
        ----------
        layer : drilldown.layer.IntervalDataLayer or drilldown.layer.PointDataLayer
            The layer used to select the points

        tolerance : float, optional
            The tolerance for identifying equivalent depths, by default 0.001.

        """
        from .layer import IntervalDataLayer, PointDataLayer

        depths = self.data[["depth"]]
        hole_id = self.data["hole ID"]

        data = layer.data

        if isinstance(layer, IntervalDataLayer):
            ranges = {}
            for hole in layer.hole_ids:
                ranges[hole] = data[data["hole ID"] == hole][["from", "to"]].values

            selected_ids = select_hole_points_by_intervals(
                hole_id, depths, ranges, tolerance
            )

        elif isinstance(layer, PointDataLayer):
            depths_2 = {}
            for hole in layer.hole_ids:
                depths_2[hole] = data[data["hole ID"] == hole]["depth"]

            selected_ids = select_hole_points_by_points(
                hole_id, depths, depths_2, tolerance
            )

        self.selected_ids = selected_ids


# Interval Mixin
class IntervalInterLayerMixin:
    """A mixin class to manager interactions between interval layers and other layers"""

    def filter_by_selection(self, layer, overlap="partial", tolerance=0.001):
        """Filter intervals by a selection performed on another layer

        If the second layer is an interval layer, the filter can be applied considering "partial" overlap, for which the intervals to which the filter is applied need only be partially overlapped by the "filtering" intervals, or "complete" overlap, for which the intervals to which the filter is applied must be completely overlapped by the "filtering" intervals.

        Parameters
        ----------
        layer : drilldown.layer.IntervalDataLayer or drilldown.layer.PointDataLayer
            The layer whose selection will be used to filter the intervals

        overlap : str, optional
            The type of overlap to consider. Must be one of "partial" or "complete", by default "partial"

        tolerance : float, optional
            The tolerance for identifying equivalent depths, by default 0.001.
        """
        from .layer import IntervalDataLayer, PointDataLayer

        from_to = self.data[["from", "to"]]
        hole_id = self.data["hole ID"]

        selected_data = layer.selected_data

        if isinstance(layer, IntervalDataLayer):
            ranges = {}
            for hole in layer.selected_hole_ids:
                ranges[hole] = selected_data[selected_data["hole ID"] == hole][
                    ["from", "to"]
                ].values

            overlap_filter = filter_hole_intervals_by_intervals(
                hole_id, from_to, ranges, overlap, tolerance
            )

        elif isinstance(layer, PointDataLayer):
            depths = {}
            for hole in layer.selected_hole_ids:
                depths[hole] = selected_data[selected_data["hole ID"] == hole]["depth"]

            overlap_filter = filter_hole_intervals_by_points(
                hole_id, from_to, depths, tolerance
            )

        self.boolean_filter = overlap_filter

    def select_by_selection(self, layer, overlap="partial", tolerance=0.001):
        """Select intervals by a selection performed on another layer

        If the second layer is an interval layer, the selection can be performed considering "partial" overlap, for which the intervals on which the selection is performed need only be partially overlapped by the "selecting" intervals, or "complete" overlap, for which the intervals on which the selection is performed must be completely overlapped by the "selecting" intervals.

        Parameters
        ----------
        layer : drilldown.layer.IntervalDataLayer or drilldown.layer.PointDataLayer
            The layer whose selection will be used to select the intervals

        overlap : str, optional
            The type of overlap to consider. Must be one of "partial" or "complete", by default "partial"

        tolerance : float, optional
            The tolerance for identifying equivalent depths, by default 0.001.

        """
        from .layer import IntervalDataLayer, PointDataLayer

        from_to = self.data[["from", "to"]]
        hole_id = self.data["hole ID"]

        selected_data = layer.selected_data

        if isinstance(layer, IntervalDataLayer):
            ranges = {}
            for hole in layer.selected_hole_ids:
                ranges[hole] = selected_data[selected_data["hole ID"] == hole][
                    ["from", "to"]
                ].values

            selected_ids = select_hole_intervals_by_intervals(
                hole_id, from_to, ranges, overlap, tolerance
            )

        elif isinstance(layer, PointDataLayer):
            depths = {}
            for hole in layer.selected_hole_ids:
                depths[hole] = selected_data[selected_data["hole ID"] == hole]["depth"]

            selected_ids = select_hole_intervals_by_points(
                hole_id, from_to, depths, tolerance
            )

        self.selected_ids = selected_ids

    def filter_by_filter(self, layer, overlap="partial", tolerance=0.001):
        """Filter intervals by a filter applied to another layer

        If the second layer is an interval layer, the filter can be applied considering "partial" overlap, for which the intervals to which the filter is applied need only be partially overlapped by the "filtering" intervals, or "complete" overlap, for which the intervals to which the filter is applied must be completely overlapped by the "filtering" intervals.

        Parameters
        ----------
        layer : drilldown.layer.IntervalDataLayer or drilldown.layer.PointDataLayer
            The layer used to filter the intervals

        overlap : str, optional
            The type of overlap to consider. Must be one of "partial" or "complete", by default "partial"

        tolerance : float, optional
            The tolerance for identifying equivalent depths, by default 0.001.

        """
        from .layer import IntervalDataLayer, PointDataLayer

        from_to = self.data[["from", "to"]]
        hole_id = self.data["hole ID"]

        filtered_data = layer.filtered_data

        if isinstance(layer, IntervalDataLayer):
            ranges = {}
            for hole in layer.filtered_hole_ids:
                ranges[hole] = filtered_data[filtered_data["hole ID"] == hole][
                    ["from", "to"]
                ].values

            overlap_filter = filter_hole_intervals_by_intervals(
                hole_id, from_to, ranges, overlap, tolerance
            )

        elif isinstance(layer, PointDataLayer):
            depths = {}
            for hole in layer.filtered_hole_ids:
                depths[hole] = filtered_data[filtered_data["hole ID"] == hole]["depth"]

            overlap_filter = filter_hole_intervals_by_points(
                hole_id, from_to, depths, tolerance
            )

        self.boolean_filter = overlap_filter

    def select_by_filter(self, layer, overlap="partial", tolerance=0.001):
        """Select intervals by a filter applied to another layer

        If the second layer is an interval layer, the selection can be performed considering "partial" overlap, for which the intervals on which the selection is performed need only be partially overlapped by the "selecting" intervals, or "complete" overlap, for which the intervals on which the selection is performed must be completely overlapped by the "selecting" intervals.

        Parameters
        ----------
        layer : drilldown.layer.IntervalDataLayer or drilldown.layer.PointDataLayer
            The layer used to select the intervals

        overlap : str, optional
            The type of overlap to consider. Must be one of "partial" or "complete", by default "partial"

        tolerance : float, optional
            The tolerance for identifying equivalent depths, by default 0.001.

        """
        from .layer import IntervalDataLayer, PointDataLayer

        from_to = self.data[["from", "to"]]
        hole_id = self.data["hole ID"]

        filtered_data = layer.filtered_data

        if isinstance(layer, IntervalDataLayer):
            ranges = {}
            for hole in layer.filtered_hole_ids:
                ranges[hole] = filtered_data[filtered_data["hole ID"] == hole][
                    ["from", "to"]
                ].values

            selected_ids = select_hole_intervals_by_intervals(
                hole_id, from_to, ranges, overlap, tolerance
            )

        elif isinstance(layer, PointDataLayer):
            depths = {}
            for hole in layer.filtered_hole_ids:
                depths[hole] = filtered_data[filtered_data["hole ID"] == hole]["depth"]

            selected_ids = select_hole_intervals_by_points(
                hole_id, from_to, depths, tolerance
            )

        self.selected_ids = selected_ids

    def filter_by(self, layer, overlap="partial", tolerance=0.001):
        """Filter intervals by another layer

        If the second layer is an interval layer, the filter can be applied considering "partial" overlap, for which the intervals to which the filter is applied need only be partially overlapped by the "filtering" intervals, or "complete" overlap, for which the intervals to which the filter is applied must be completely overlapped by the "filtering" intervals.

        Parameters
        ----------
        layer : drilldown.layer.IntervalDataLayer or drilldown.layer.PointDataLayer
            The layer used to filter the intervals

        overlap : str, optional
            The type of overlap to consider. Must be one of "partial" or "complete", by default "partial"

        tolerance : float, optional
            The tolerance for identifying equivalent depths, by default 0.001.

        """
        from .layer import IntervalDataLayer, PointDataLayer

        from_to = self.data[["from", "to"]]
        hole_id = self.data["hole ID"]

        data = layer.data

        if isinstance(layer, IntervalDataLayer):
            ranges = {}
            for hole in layer.hole_ids:
                ranges[hole] = data[data["hole ID"] == hole][["from", "to"]].values

            overlap_filter = filter_hole_intervals_by_intervals(
                hole_id, from_to, ranges, overlap, tolerance
            )

        elif isinstance(layer, PointDataLayer):
            depths = {}
            for hole in layer.hole_ids:
                depths[hole] = data[data["hole ID"] == hole]["depth"]

            overlap_filter = filter_hole_intervals_by_points(
                hole_id, from_to, depths, tolerance
            )

        self.boolean_filter = overlap_filter

    def select_by(self, layer, overlap="partial", tolerance=0.001):
        """Select intervals by another layer

        If the second layer is an interval layer, the selection can be performed considering "partial" overlap, for which the intervals on which the selection is performed need only be partially overlapped by the "selecting" intervals, or "complete" overlap, for which the intervals on which the selection is performed must be completely overlapped by the "selecting" intervals.

        Parameters
        ----------
        layer : drilldown.layer.IntervalDataLayer or drilldown.layer.PointDataLayer
            The layer used to select the intervals

        overlap : str, optional
            The type of overlap to consider. Must be one of "partial" or "complete", by default "partial"

        tolerance : float, optional
            The tolerance for identifying equivalent depths, by default 0.001.

        """
        from .layer import IntervalDataLayer, PointDataLayer

        from_to = self.data[["from", "to"]]
        hole_id = self.data["hole ID"]

        data = layer.data

        if isinstance(layer, IntervalDataLayer):
            ranges = {}
            for hole in layer.hole_ids:
                ranges[hole] = data[data["hole ID"] == hole][["from", "to"]].values

            selected_ids = select_hole_intervals_by_intervals(
                hole_id, from_to, ranges, overlap, tolerance
            )

        elif isinstance(layer, PointDataLayer):
            depths = {}
            for hole in layer.hole_ids:
                depths[hole] = data[data["hole ID"] == hole]["depth"]

            selected_ids = select_hole_intervals_by_points(
                hole_id, from_to, depths, tolerance
            )

        self.selected_ids = selected_ids

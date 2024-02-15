import pandas as pd


# intervals by intervals
def filter_intervals_by_intervals(from_to, ranges, overlap="partial", tolerance=0.001):
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
    overlap_filter = filter_intervals_by_intervals(from_to, ranges, overlap, tolerance)
    overlap_selection = from_to[overlap_filter].index.tolist()

    return overlap_selection


def filter_hole_intervals_by_intervals(
    hole_id, from_to, ranges, overlap="partial", tolerance=0.001
):
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
    overlap_fun = lambda row: any(
        ((depth - row["from"] > tolerance) and (row["to"] - depth > tolerance))
        for depth in depths
    )

    overlap_filter = from_to.apply(overlap_fun, axis=1).tolist()

    return overlap_filter


def select_intervals_by_points(from_to, depths, tolerance=0.001):
    overlap_filter = filter_intervals_by_points(from_to, depths, tolerance)
    overlap_selection = from_to[overlap_filter].index.tolist()

    return overlap_selection


def filter_hole_intervals_by_points(hole_id, from_to, depths, tolerance=0.001):
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
    selected_ids = []
    hole_ids = [hole for hole in list(set(hole_id)) if hole in depths.keys()]
    for hole in hole_ids:
        hole_filter = hole_id == hole
        hole_from_to = from_to[hole_filter]
        hole_depths = depths[hole]
        selected_ids += select_intervals_by_points(hole_from_to, hole_depths, tolerance)

    return selected_ids


# Interval Mixin
class IntervalInterLayerMixin:
    def filter_by_selection(self, layer, overlap="partial", tolerance=0.001):
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

        return self

    def select_by_selection(self, layer, overlap="partial", tolerance=0.001):
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

        return self

    def filter_by_filter(self, layer, overlap="partial", tolerance=0.001):
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

        return self

    def select_by_filter(self, layer, overlap="partial", tolerance=0.001):
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

        return self

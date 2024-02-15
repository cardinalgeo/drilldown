import pandas as pd


def filter_by_depth_ranges(from_to, ranges, overlap="partial"):
    partial_fun = lambda row: any(
        (not ((row["from"] >= to_depth) or (row["to"] <= from_depth)))
        for from_depth, to_depth in ranges
    )
    complete_fun = lambda row: any(
        (
            (not ((row["from"] >= to_depth) or (row["to"] <= from_depth)))
            and (row["from"] >= from_depth)
            and (row["to"] <= to_depth)
        )
        for from_depth, to_depth in ranges
    )

    if overlap == "complete":
        overlap_fun = complete_fun
    elif overlap == "partial":
        overlap_fun = partial_fun

    overlap_filter = from_to.apply(overlap_fun, axis=1).tolist()

    return overlap_filter


def select_by_depth_ranges(from_to, ranges, overlap="partial"):
    overlap_filter = filter_by_depth_ranges(from_to, ranges, overlap)
    overlap_selection = from_to[overlap_filter].index.tolist()

    return overlap_selection


def filter_holes_by_depth_ranges(hole_id, from_to, ranges, overlap="partial"):
    overlap_filter = pd.Series(index=hole_id.index, data=False)
    hole_ids = [hole for hole in list(set(hole_id)) if hole in ranges.keys()]
    for hole in hole_ids:
        hole_filter = hole_id == hole
        hole_from_to = from_to[hole_filter]
        hole_ranges = ranges[hole]
        overlap_filter[hole_filter] = filter_by_depth_ranges(
            hole_from_to, hole_ranges, overlap
        )
        print(overlap_filter[hole_filter])

    return overlap_filter.tolist()


def select_holes_by_depth_ranges(hole_id, from_to, ranges, overlap="partial"):
    selected_ids = []
    hole_ids = [hole for hole in list(set(hole_id)) if hole in ranges.keys()]
    for hole in hole_ids:
        hole_filter = hole_id == hole
        hole_from_to = from_to[hole_filter]
        hole_ranges = ranges[hole]
        selected_ids += select_by_depth_ranges(hole_from_to, hole_ranges, overlap)

    return selected_ids


class IntervalByMixin:
    def filter_by_selection(self, layer, overlap="partial"):
        from_to = self.data[["from", "to"]]
        hole_id = self.data["hole ID"]

        selected_data = layer.selected_data
        ranges = {}
        for hole in layer.selected_hole_ids:
            ranges[hole] = selected_data[selected_data["hole ID"] == hole][
                ["from", "to"]
            ].values

        overlap_filter = filter_holes_by_depth_ranges(hole_id, from_to, ranges, overlap)
        self.boolean_filter = overlap_filter

        return self

    def select_by_selection(self, layer, overlap="partial"):
        from_to = self.data[["from", "to"]]
        hole_id = self.data["hole ID"]

        selected_data = layer.selected_data
        ranges = {}
        for hole in layer.selected_hole_ids:
            ranges[hole] = selected_data[selected_data["hole ID"] == hole][
                ["from", "to"]
            ].values

        selected_ids = select_holes_by_depth_ranges(hole_id, from_to, ranges, overlap)
        self.selected_ids = selected_ids

        return self

    def filter_by_filter(self, layer, overlap="partial"):
        from_to = self.data[["from", "to"]]
        hole_id = self.data["hole ID"]

        filtered_data = layer.filtered_data
        ranges = {}
        for hole in layer.filtered_hole_ids:
            ranges[hole] = filtered_data[filtered_data["hole ID"] == hole][
                ["from", "to"]
            ].values

        overlap_filter = filter_holes_by_depth_ranges(hole_id, from_to, ranges, overlap)
        self.boolean_filter = overlap_filter

        return self

    def select_by_filter(self, layer, overlap="partial"):
        from_to = self.data[["from", "to"]]
        hole_id = self.data["hole ID"]

        filtered_data = layer.filtered_data
        ranges = {}
        for hole in layer.filtered_hole_ids:
            ranges[hole] = filtered_data[filtered_data["hole ID"] == hole][
                ["from", "to"]
            ].values

        selected_ids = select_holes_by_depth_ranges(hole_id, from_to, ranges, overlap)
        self.selected_ids = selected_ids

        return self

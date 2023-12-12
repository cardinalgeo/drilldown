import pyvista as pv
from geoh5py.workspace import Workspace
from geoh5py.groups import DrillholeGroup
from geoh5py.objects import Drillhole
import numpy as np
import pandas as pd
import distinctipy

from .plotter import DrillDownPlotter
from .drill_log import DrillLog

from matplotlib.colors import ListedColormap


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


def make_matplotlib_categorical_color_map(colors):
    mapping = np.linspace(0, len(colors) - 1, 256)
    new_colors = np.empty((256, 3))
    i_pre = -np.inf
    for i, color in enumerate(colors):
        new_colors[(mapping > i_pre) & (mapping <= i)] = list(color)
        i_pre = i
    map = ListedColormap(colors[0:-1])
    map.set_under(colors[0])
    map.set_over(colors[-1])
    return map


def make_color_map_fractional(map):
    for name in map.keys():
        if max(map[name]) > 1:
            map[name] = tuple([val / 255 for val in map[name]])
    return map


def encode_categorical_data(data):
    data_encoded, categories = pd.factorize(data)
    codes = np.arange(len(categories))

    # convert codes to float
    codes = np.array(codes, dtype="float")
    data_encoded = np.array(data_encoded, dtype="float")

    # center numerical representation of categorical data, while maintaining range, to address pyvista's color mapping querks
    codes[1:-1] += 0.5
    data_encoded[
        (data_encoded != data_encoded.min()) & (data_encoded != data_encoded.max())
    ] += 0.5
    code_to_cat_map = {code: cat for code, cat in zip(codes, categories)}

    return code_to_cat_map, data_encoded


class HoleData:
    def __init__(self):
        self.hole_ids = []
        self.vars_all = []
        self.categorical_vars = []
        self.continuous_vars = []
        self._depths = None
        self.data = {}
        self.code_to_color_map = {}
        self.cat_to_color_map = {}
        self.code_to_cat_map = {}
        self.cat_to_code_map = {}
        self.matplotlib_formatted_color_maps = {}
        self.categorical_color_rng = 999
        self.categorical_pastel_factor = 0.2

    def add_data(
        self,
        var_names,
        hole_ids,
        depths,
        data,
        return_data=False,
        construct_categorical_cmap=False,
    ):
        # add vars
        self.vars_all += var_names

        # save flag to construct categorical color map
        self.construct_categorical_cmap = construct_categorical_cmap

        # add hole IDs
        if isinstance(hole_ids, pd.core.series.Series):
            hole_ids = hole_ids.values
        self.hole_ids = hole_ids

        # encode hole IDs, as strings are wiped in pyvista meshes
        self.hole_ids_encoded, hole_ids_unique = pd.factorize(hole_ids)
        self.hole_id_to_code_map = {
            hole_id: code for code, hole_id in enumerate(hole_ids_unique)
        }
        self.code_to_hole_id_map = {
            code: hole_id for code, hole_id in enumerate(hole_ids_unique)
        }

        # add from-to depths
        self.depths = depths

        # add data
        if (isinstance(data, pd.core.series.Series)) | (
            isinstance(data, pd.core.frame.DataFrame)
        ):
            data = data.values
        for dataset, var_name in zip(data.T, var_names):
            dataset, _type = convert_array_type(dataset, return_type=True)

            if _type == "str":  # categorical data
                self.categorical_vars.append(var_name)

                # encode categorical data
                code_to_cat_map, dataset = encode_categorical_data(dataset)
                self.code_to_cat_map[var_name] = code_to_cat_map
                self.cat_to_code_map[var_name] = {
                    cat: code for code, cat in code_to_cat_map.items()
                }

            else:
                self.continuous_vars.append(var_name)

            self.data[var_name] = {
                "values": dataset,
                "type": _type,
            }

        if return_data == True:
            return self.data

    def _construct_categorical_cmap(self):
        var_names = [
            var
            for var in self.categorical_vars
            if var not in self.cat_to_color_map.keys()
        ]
        for var in var_names:
            codes = self.code_to_cat_map[var].keys()
            n_colors = len(codes)

            colors = distinctipy.get_colors(
                n_colors,
                pastel_factor=self.categorical_pastel_factor,
                rng=self.categorical_color_rng,
            )

            # create categorical color map
            self.cat_to_color_map[var] = {
                cat: color
                for cat, color in zip(self.cat_to_code_map[var].keys(), colors)
            }

            # create encoded categorical color map
            self.code_to_color_map[var] = {
                code: color for code, color in zip(codes, colors)
            }

            # create matplotlib categorical color map
            self.matplotlib_formatted_color_maps[
                var
            ] = make_matplotlib_categorical_color_map(colors)

    def add_categorical_cmap(self, var_name, cmap):
        # ensure categorical color map colors are fractional
        cmap = make_color_map_fractional(cmap)

        colors = [cmap[cat] for cat in cmap.keys()]
        n_missing_colors = len(self.cat_to_code_map[var_name].keys()) - len(colors)
        colors = distinctipy.get_colors(
            n_missing_colors,
            exclude_colors=colors,
            return_excluded=True,
            pastel_factor=self.categorical_pastel_factor,
            rng=self.categorical_color_rng,
        )

        # create categorical color map
        categories = [cat for cat in cmap.keys()] + [
            cat
            for cat in self.cat_to_code_map[var_name].keys()
            if cat not in cmap.keys()
        ]
        self.cat_to_color_map[var_name] = {
            cat: color for cat, color in zip(categories, colors)
        }

        # create encoded categorical color map
        codes = [self.cat_to_code_map[var_name][cat] for cat in categories]
        self.code_to_color_map[var_name] = {
            code: color for code, color in zip(codes, colors)
        }

        # create matplotlib categorical color map
        codes.sort()
        self.matplotlib_formatted_color_maps[
            var_name
        ] = make_matplotlib_categorical_color_map(
            [self.code_to_color_map[var_name][code] for code in codes]
        )

    @property
    def depths(self):
        return self._depths

    @depths.setter
    def depths(self, depths):
        if isinstance(depths, pd.core.frame.DataFrame):
            depths = depths.values
        self._depths = depths.astype(np.float64)


class Points(HoleData):
    def __init__(self):
        super().__init__()

    def add_data(
        self,
        var_names,
        hole_ids,
        depths,
        data,
        return_data=False,
        construct_categorical_cmap=False,
    ):
        super().add_data(
            var_names,
            hole_ids,
            depths,
            data,
            return_data=return_data,
            construct_categorical_cmap=construct_categorical_cmap,
        )

    def drill_log(self, hole_id, categorical_vars=None, continuous_vars=None):
        if self.construct_categorical_cmap == True:
            # ensure that color maps exist for categorical vars
            self._construct_categorical_cmap()

        log = DrillLog()

        if categorical_vars is None:
            categorical_vars = self.categorical_vars

        if continuous_vars is None:
            continuous_vars = self.continuous_vars

        depths = self.depths[self.hole_ids == hole_id]

        for var in categorical_vars:
            values = self.data[var]["values"][self.hole_ids == hole_id]
            cat_to_color_map = self.cat_to_color_map.get(var, None)
            log.add_categorical_point_data(var, depths, values, cat_to_color_map)

        for var in continuous_vars:
            values = self.data[var]["values"][self.hole_ids == hole_id]

            log.add_continuous_point_data(var, depths, values)

        log.create_figure(y_axis_label="Depth (m)", title=hole_id)

        return log.fig


class Intervals(HoleData):
    def __init__(self):
        super().__init__()

    def add_data(
        self,
        var_names,
        hole_ids,
        depths,
        data,
        return_data=False,
        construct_categorical_cmap=True,
    ):
        super().add_data(
            var_names,
            hole_ids,
            depths,
            data,
            return_data=return_data,
            construct_categorical_cmap=construct_categorical_cmap,
        )

    def drill_log(self, hole_id, categorical_vars=None, continuous_vars=None):
        if self.construct_categorical_cmap == True:
            # ensure that color maps exist for categorical vars
            self._construct_categorical_cmap()

        log = DrillLog()

        if categorical_vars is None:
            categorical_vars = self.categorical_vars

        if continuous_vars is None:
            continuous_vars = self.continuous_vars

        from_to = self.depths[self.hole_ids == hole_id]

        for var in categorical_vars:
            values = self.data[var]["values"][self.hole_ids == hole_id]
            values = np.array([self.code_to_cat_map[var][val] for val in values])
            cat_to_color_map = self.cat_to_color_map.get(var, None)
            log.add_categorical_interval_data(var, from_to, values, cat_to_color_map)

        for var in continuous_vars:
            values = self.data[var]["values"][self.hole_ids == hole_id]

            log.add_continuous_interval_data(var, from_to, values)

        log.create_figure(y_axis_label="Depth (m)", title=hole_id)

        return log.fig


class DrillHole:
    def __init__(self):
        self.workspace = Workspace()
        self.hole_group = DrillholeGroup.create(self.workspace)
        self.vars = []
        self.categorical_interval_vars = []
        self.continuous_interval_vars = []
        self.categorical_point_vars = []
        self.continuous_point_vars = []

        self.intervals = {}
        self.points = {}

        self.cat_to_code_map = {}
        self.code_to_cat_map = {}
        self.code_to_color_map = {}
        self.cat_to_color_map = {}
        self.matplotlib_formatted_color_maps = {}

    def add_collar(self, collar):
        if isinstance(collar, pd.core.series.Series) | isinstance(
            collar, pd.core.frame.DataFrame
        ):
            collar = collar.values[0]

        self.collar = collar

    def add_survey(self, dist, azm, dip):
        if isinstance(dist, pd.core.series.Series):
            dist = dist.values
        if isinstance(azm, pd.core.series.Series):
            azm = azm.values
        if isinstance(dip, pd.core.series.Series):
            dip = dip.values

        self.survey = np.c_[dist, azm, dip]
        self._create_hole()

    def _create_hole(self):
        self._hole = Drillhole.create(
            self.workspace,
            collar=self.collar,
            surveys=self.survey,
            name="",
            parent=self.hole_group,
        )

    def _add_from_to(self, from_to):
        if isinstance(from_to, pd.core.frame.DataFrame):
            from_to = from_to.values
        self.from_to = from_to.astype(np.float64)

        return self.from_to

    def _make_from_to(self, depths, connected=True):
        if connected == True:
            from_depths = depths[:-1]
            to_depths = depths[1:]

            if depths.ndim == 1:
                depths = np.empty([from_depths.shape[0], 2])
            else:
                depths = np.empty([from_depths.shape[0], 2, from_depths.shape[1]])

            depths[:, 0] = from_depths
            depths[:, 1] = to_depths

            return depths

    def _add_data(self, name, data):
        self.cat_to_code_map[name] = data.cat_to_code_map
        self.code_to_cat_map[name] = data.code_to_cat_map
        self.code_to_color_map[name] = data.code_to_color_map
        self.cat_to_color_map[name] = data.cat_to_color_map
        self.matplotlib_formatted_color_maps[
            name
        ] = data.matplotlib_formatted_color_maps

    def add_intervals(self, name, intervals):
        self.intervals[name] = intervals
        self.categorical_interval_vars += intervals.categorical_vars
        self.continuous_interval_vars += intervals.continuous_vars
        self.vars += intervals.vars_all
        self._add_data(name, intervals)

    def add_points(self, name, points):
        self.points[name] = points
        self.categorical_point_vars += points.categorical_vars
        self.continuous_point_vars += points.continuous_vars
        self.vars += points.vars_all
        self._add_data(name, points)

    def desurvey(self, depths=None):
        if depths is None:
            # return desurveyed survey depths if no depths passed
            return self._hole.desurvey(self.survey[:, 0])
        else:
            return self._hole.desurvey(depths)

    def _make_line_mesh(self, from_depth, to_depth):
        """Make a mesh consisting of line segments for which a connected topology is assumed."""
        depths = np.empty((from_depth.shape[0] + to_depth.shape[0], 3))
        depths[0::2, :] = from_depth
        depths[1::2, :] = to_depth
        n_connected = np.ones(int(depths.shape[0] / 2), dtype="int") * 2
        from_positions = np.arange(0, depths.shape[0] - 1, 2)
        to_positions = np.arange(1, depths.shape[0], 2)
        depth_connectivity = np.hstack(
            np.stack([n_connected, from_positions, to_positions], axis=1)
        )
        mesh = pv.PolyData(depths, lines=depth_connectivity)

        return mesh

    def make_collar_mesh(self):
        mesh = pv.PolyData(self.collar)

        return mesh

    def make_survey_mesh(self):
        depths = self.desurvey()
        from_to = self._make_from_to(depths)
        mesh = self._make_line_mesh(from_to[0], from_to[1])

        return mesh

    def make_intervals_mesh(self, name=None):
        if name is None:
            name = list(self.intervals.keys())[0]

        intervals = self.intervals[name]
        from_to = intervals.depths
        from_depths = self.desurvey(from_to[:, 0])
        to_depths = self.desurvey(from_to[:, 1])
        intermediate_depths = np.mean([from_depths, to_depths], axis=0)
        mesh = self._make_line_mesh(from_depths, to_depths)

        mesh.cell_data["from"] = from_to[:, 0]
        mesh.cell_data["to"] = from_to[:, 1]
        mesh.cell_data["x"] = intermediate_depths[:, 0]
        mesh.cell_data["y"] = intermediate_depths[:, 1]
        mesh.cell_data["z"] = intermediate_depths[:, 2]
        for var in intervals.vars_all:
            data = intervals.data[var]["values"]
            _type = intervals.data[var]["type"]
            if _type == "str":
                mesh[var] = data
            else:
                mesh.cell_data[var] = data

        return mesh

    def make_points_mesh(self, name=None):
        if name is None:
            name = list(self.points.keys())[0]

        points = self.points[name]
        depths = points.depths
        depths = self.desurvey(depths)
        mesh = pv.PolyData(depths)
        for var in points.vars_all:
            data = points.data[var]["values"]
            _type = points.data[var]["type"]
            if _type == "str":
                mesh[var] = data
            else:
                mesh.point_data[var] = data

        return mesh

    def show_collar(self, *args, **kwargs):
        collar_mesh = self.make_collar_mesh()
        p = DrillDownPlotter()
        p.add_collars(collar_mesh, *args, **kwargs)

        return p.show()

    def show_survey(self, show_collar=True, *args, **kwargs):
        survey_mesh = self.make_survey_mesh()
        p = DrillDownPlotter()
        p.add_surveys(survey_mesh, *args, **kwargs)

        if show_collar == True:
            collar_mesh = self.make_collar_mesh()
            p.add_collars(collar_mesh)

        return p.show()

    def show_intervals(
        self, name=None, show_collar=True, show_survey=True, *args, **kwargs
    ):
        if name is None:
            name = list(self.intervals.keys())[0]

        intervals_mesh = self.make_intervals_mesh(name)
        p = DrillDownPlotter()
        p.add_intervals(
            name,
            intervals_mesh,
            self.categorical_interval_vars,
            self.continuous_interval_vars,
            *args,
            **kwargs
        )

        if show_collar == True:
            collar_mesh = self.make_collar_mesh()
            p.add_collars(collar_mesh)

        if show_survey == True:
            survey_mesh = self.make_survey_mesh()
            p.add_surveys(survey_mesh)

        return p.show()

    def show_points(
        self, name=None, show_collar=True, show_survey=True, *args, **kwargs
    ):
        if name is None:
            name = list(self.points.keys())[0]

        points_mesh = self.make_points_mesh(name)
        p = DrillDownPlotter()
        p.add_points(
            name,
            points_mesh,
            self.categorical_point_vars,
            self.continuous_point_vars,
            *args,
            **kwargs
        )

        if show_collar == True:
            collar_mesh = self.make_collar_mesh()
            p.add_collars(collar_mesh)

        if show_survey == True:
            survey_mesh = self.make_survey_mesh()
            p.add_surveys(survey_mesh)

        return p.show()

    def show(self):
        collar_mesh = self.make_collar_mesh()
        survey_mesh = self.make_survey_mesh()
        intervals_mesh = self.make_intervals_mesh()
        points_mesh = self.make_points_mesh()

        p = DrillDownPlotter()
        p.add_collars(collar_mesh)
        p.add_surveys(survey_mesh)

        intervals_name = list(self.intervals.keys())[0]
        p.add_intervals(
            intervals_name,
            intervals_mesh,
            self.categorical_interval_vars,
            self.continuous_interval_vars,
            radius=10,
        )

        points_name = list(self.points.keys())[0]
        p.add_points(
            points_name,
            points_mesh,
            self.categorical_point_vars,
            self.continuous_point_vars,
        )

        return p.show()

    def drill_log(
        self,
        categorical_interval_vars=[],
        continuous_interval_vars=[],
        categorical_point_vars=[],
        continuous_point_vars=[],
    ):
        if self.intervals or self.points:  # ensure there is data to plot
            log = DrillLog()

            # check if no variables are passed; if so, use all variables
            if any(
                len(_) != 0
                for _ in [
                    categorical_interval_vars,
                    continuous_interval_vars,
                    categorical_point_vars,
                    continuous_point_vars,
                ]
            ):
                pass

            else:
                categorical_interval_vars = self.categorical_interval_vars
                continuous_interval_vars = self.continuous_interval_vars
                categorical_point_vars = self.categorical_point_vars
                continuous_point_vars = self.continuous_point_vars

            # add interval data
            if self.intervals:
                name = list(self.intervals.keys())[0]
                intervals = self.intervals[name]
                from_to = intervals.depths

                for var in categorical_interval_vars:
                    cat_to_color_map = self.cat_to_color_map[name]
                    values = intervals.data[var]["values"]
                    values = np.array(
                        [self.code_to_cat_map[name][var][val] for val in values]
                    )
                    log.add_categorical_interval_data(
                        var, from_to, values, cat_to_color_map.get(var, None)
                    )

                for var in continuous_interval_vars:
                    values = intervals.data[var]["values"]

                    log.add_continuous_interval_data(var, from_to, values)

            # add point data
            if self.points:
                name = list(self.points.keys())[0]
                points = name
                depths = points.depths

                for var in categorical_point_vars:
                    cat_to_color_map = self.cat_to_color_map[name]
                    values = points.data[var]["values"]
                    values = np.array(
                        [self.code_to_cat_map[name][var][val] for val in values]
                    )
                    log.add_categorical_point_data(
                        var, depths, values, cat_to_color_map.get(var, None)
                    )

                for var in continuous_point_vars:
                    pass

            log.create_figure(y_axis_label="Depth (m)")

            return log.fig


class DrillHoleGroup:
    def __init__(self):
        self._holes = {}
        self.vars = []
        self.categorical_interval_vars = []
        self.continuous_interval_vars = []
        self.categorical_point_vars = []
        self.continuous_point_vars = []

        self.intervals = {}
        self.points = {}

        self.workspace = Workspace()
        self.hole_ids_with_data = []

        self.hole_id_to_code_map = {}
        self.code_to_hole_id_map = {}

        self.cat_to_code_map = {}
        self.code_to_cat_map = {}
        self.code_to_color_map = {}
        self.cat_to_color_map = {}
        self.matplotlib_formatted_color_maps = {}

    def add_collars(self, hole_id, collars):
        if isinstance(hole_id, pd.core.series.Series):
            hole_id = hole_id.values

        if isinstance(collars, pd.core.frame.DataFrame):
            collars = collars.values

        self.hole_ids = np.unique(hole_id)
        self.collars = np.c_[hole_id, collars]

    def add_surveys(self, hole_id, dist, azm, dip):
        if isinstance(hole_id, pd.core.series.Series):
            hole_id = hole_id.values

        if isinstance(dist, pd.core.series.Series):
            dist = dist.values

        if isinstance(azm, pd.core.series.Series):
            azm = azm.values

        if isinstance(dip, pd.core.series.Series):
            dip = dip.values

        self.surveys = np.c_[hole_id, dist, azm, dip]

        if self.collars is not None:
            self._create_holes()

    def _create_holes(self):
        for hole_id in self.hole_ids:
            hole = DrillHole()

            hole.add_collar(self.collars[self.collars[:, 0] == hole_id, 1:][0])

            surveys = np.hsplit(self.surveys[self.surveys[:, 0] == hole_id, 1:], 3)
            if (surveys[0].shape[0]) > 0:
                hole.add_survey(surveys[0], surveys[1], surveys[2])

                hole._create_hole()

                self._holes[hole_id] = hole

    def _add_from_to(self, hole_ids, from_to):
        if isinstance(hole_ids, pd.core.series.Series):
            hole_ids = hole_ids.values

        if isinstance(from_to, pd.core.frame.DataFrame):
            from_to = from_to.values

        self.from_to = np.c_[hole_ids, from_to]
        hole_ids = [id for id in np.unique(hole_ids) if id in self._holes.keys()]
        for id in hole_ids:
            dataset = self.from_to[self.from_to[:, 0] == id, 1:]
            if dataset.shape[0] > 0:
                self.hole_ids_with_data.append(id)
                self._holes[id]._add_from_to(dataset)

        return self.from_to

    def _add_data(self, name, data):
        self.hole_ids_with_data += list(np.unique(data.hole_ids))
        self.hole_id_to_code_map[name] = data.hole_id_to_code_map
        self.code_to_hole_id_map[name] = data.code_to_hole_id_map
        self.cat_to_code_map[name] = data.cat_to_code_map
        self.code_to_cat_map[name] = data.code_to_cat_map
        self.code_to_color_map[name] = data.code_to_color_map
        self.cat_to_color_map[name] = data.cat_to_color_map
        self.matplotlib_formatted_color_maps[
            name
        ] = data.matplotlib_formatted_color_maps

    def add_intervals(self, name, intervals):
        self.intervals[name] = intervals
        self.categorical_interval_vars += intervals.categorical_vars
        self.continuous_interval_vars += intervals.continuous_vars
        self.vars += intervals.vars_all
        self._add_data(name, intervals)

    def add_points(self, name, points):
        self.points[name] = points
        self.categorical_point_vars += points.categorical_vars
        self.continuous_point_vars += points.continuous_vars
        self.vars += points.vars_all
        self._add_data(name, points)

    def desurvey(self, hole_id, depths=None):
        return self._holes[hole_id].desurvey(depths=depths)

    def make_collars_mesh(self):
        mesh = pv.PolyData(np.asarray(self.collars[:, 1:], dtype="float"))

        return mesh

    def make_surveys_mesh(self):
        mesh = None
        for hole_id in self._holes.keys():
            hole = self._holes[hole_id]
            depths = hole.desurvey()
            from_to = hole._make_from_to(depths)
            if from_to.shape[0] > 0:
                if mesh is None:
                    mesh = hole._make_line_mesh(from_to[:, 0], from_to[:, 1])
                else:
                    mesh += hole._make_line_mesh(from_to[:, 0], from_to[:, 1])

        return mesh

    def make_intervals_mesh(self, name=None):
        if name is None:
            name = list(self.intervals.keys())[0]

        intervals = self.intervals[name]
        meshes = None
        for id in self.hole_ids_with_data:
            from_to = intervals.depths[intervals.hole_ids == id]
            hole = self._holes[id]
            from_depths = hole.desurvey(from_to[:, 0])
            to_depths = hole.desurvey(from_to[:, 1])
            intermediate_depths = np.mean([from_depths, to_depths], axis=0)
            if from_depths.shape[0] > 0:
                mesh = hole._make_line_mesh(from_depths, to_depths)

                mesh.cell_data["from"] = from_to[:, 0]
                mesh.cell_data["to"] = from_to[:, 1]
                mesh.cell_data["hole ID"] = [
                    intervals.hole_id_to_code_map[id]
                ] * from_to.shape[0]
                mesh.cell_data["x"] = intermediate_depths[:, 0]
                mesh.cell_data["y"] = intermediate_depths[:, 1]
                mesh.cell_data["z"] = intermediate_depths[:, 2]
                for var in intervals.vars_all:
                    data = intervals.data[var]["values"][intervals.hole_ids == id]
                    _type = intervals.data[var]["type"]
                    if _type == "str":
                        mesh[var] = data
                    else:
                        mesh.cell_data[var] = data
                if meshes is None:
                    meshes = mesh
                else:
                    meshes += mesh

        return meshes

    def make_points_mesh(self, name=None):
        if name is None:
            name = list(self.points.keys())[0]

        points = self.points[name]
        meshes = None
        for id in self.hole_ids_with_data:
            depths = points.depths[points.hole_ids == id]
            hole = self._holes[id]
            depths = hole.desurvey(depths)
            if depths.shape[0] > 0:
                mesh = pv.PolyData(depths)
                for var in points.vars_all:
                    data = points.data[var]["values"][points.hole_ids == id]
                    _type = points.data[var]["type"]
                    if _type == "str":
                        mesh[var] = data
                    else:
                        mesh.point_data[var] = data
                mesh.point_data["hole ID"] = [
                    points.hole_id_to_code_map[id]
                ] * depths.shape[0]
                if meshes is None:
                    meshes = mesh
                else:
                    meshes += mesh

        return meshes

    def show_collars(self):
        mesh = self.make_collars_mesh()

        return mesh.plot()

    def show_surveys(self):
        mesh = self.make_surveys_mesh()

        return mesh.plot()

    def show_intervals(self, name=None):
        mesh = self.make_intervals_mesh(name)

        return mesh.plot()

    def show_points(self, name=None):
        mesh = self.make_surveys_mesh(name)

        return mesh.plot()

    def show_collars(self, *args, **kwargs):
        collars_mesh = self.make_collars_mesh()
        p = DrillDownPlotter()
        p.add_collars(collars_mesh, *args, **kwargs)

        return p.show()

    def show_surveys(self, show_collars=True, *args, **kwargs):
        surveys_mesh = self.make_surveys_mesh()
        p = DrillDownPlotter()
        p.add_surveys(surveys_mesh, *args, **kwargs)

        if show_collars == True:
            collars_mesh = self.make_collars_mesh()
            p.add_collars(collars_mesh)

        return p.show()

    def show_intervals(
        self, name=None, show_collars=True, show_surveys=True, *args, **kwargs
    ):
        if name is None:
            name = list(self.intervals.keys())[0]

        intervals_mesh = self.make_intervals_mesh(name)
        p = DrillDownPlotter()
        p.add_intervals(
            name,
            intervals_mesh,
            self.categorical_point_vars,
            self.continuous_point_vars,
            *args,
            **kwargs
        )

        if show_collars == True:
            collars_mesh = self.make_collars_mesh()
            p.add_collars(collars_mesh)

        if show_surveys == True:
            surveys_mesh = self.make_surveys_mesh()
            p.add_surveys(surveys_mesh)

        return p.show()

    def show_points(
        self, name=None, show_collars=True, show_surveys=True, *args, **kwargs
    ):
        if name is None:
            name = list(self.points.keys())[0]

        points_mesh = self.make_points_mesh(name)
        p = DrillDownPlotter()
        p.add_points(
            name,
            points_mesh,
            self.categorical_point_vars,
            self.continuous_point_vars,
            *args,
            **kwargs
        )

        if show_collars == True:
            collars_mesh = self.make_collars_mesh()
            p.add_collars(collars_mesh)

        if show_surveys == True:
            surveys_mesh = self.make_surveys_mesh()
            p.add_surveys(surveys_mesh)

        return p.show()

    def show(self):
        p = DrillDownPlotter()

        # add color-category and code-category maps
        p.code_to_hole_id_map = self.code_to_hole_id_map
        p.hole_id_to_code_map = self.hole_id_to_code_map
        p.code_to_cat_map = self.code_to_cat_map
        p.cat_to_code_map = self.cat_to_code_map
        p.code_to_color_map = self.code_to_color_map
        p.cat_to_color_map = self.cat_to_color_map
        p.matplotlib_formatted_color_maps = self.matplotlib_formatted_color_maps
        # create and add collars mesh
        collars_mesh = self.make_collars_mesh()
        p.add_collars(collars_mesh)

        # create and add surveys mesh
        surveys_mesh = self.make_surveys_mesh()
        p.add_surveys(surveys_mesh)

        # create and add intervals mesh(es)
        for name in self.intervals.keys():
            intervals_mesh = self.make_intervals_mesh(name)
            p.add_intervals(
                name,
                intervals_mesh,
                self.categorical_interval_vars,
                self.continuous_interval_vars,
            )

        # create and add points mesh(es)
        for name in self.points.keys():
            points_mesh = self.make_points_mesh(name)
            p.add_points(
                name,
                points_mesh,
                self.categorical_point_vars,
                self.continuous_point_vars,
            )

        return p.show()

    def drill_log(
        self,
        hole_id,
        categorical_interval_vars=[],
        continuous_interval_vars=[],
        categorical_point_vars=[],
        continuous_point_vars=[],
    ):
        if self.intervals or self.points:  # ensure there is data to plot
            log = DrillLog()

            # check if no variables are passed; if so, use all variables
            if any(
                len(_) != 0
                for _ in [
                    categorical_interval_vars,
                    continuous_interval_vars,
                    categorical_point_vars,
                    continuous_point_vars,
                ]
            ):
                pass

            else:
                categorical_interval_vars = self.categorical_interval_vars
                continuous_interval_vars = self.continuous_interval_vars
                categorical_point_vars = self.categorical_point_vars
                continuous_point_vars = self.continuous_point_vars

            # add interval data
            if self.intervals:
                name = list(self.intervals.keys())[0]
                intervals = self.intervals[name]
                from_to = intervals.depths[intervals.hole_ids == hole_id]

                for var in categorical_interval_vars:
                    cat_to_color_map = self.cat_to_color_map[name]
                    values = intervals.data[var]["values"][
                        intervals.hole_ids == hole_id
                    ]
                    values = np.array(
                        [self.code_to_cat_map[name][var][val] for val in values]
                    )
                    log.add_categorical_interval_data(
                        var,
                        from_to,
                        values,
                        cat_to_color_map.get(var, None),
                    )

                for var in continuous_interval_vars:
                    values = intervals.data[var]["values"][
                        intervals.hole_ids == hole_id
                    ]

                    log.add_continuous_interval_data(var, from_to, values)

            # add point data
            if self.points:
                name = list(self.points.keys())[0]
                points = self.points[name]
                depths = points.depths[points.hole_ids == hole_id]

                for var in categorical_point_vars:
                    cat_to_color_map = self.cat_to_color_map[name]
                    values = points.data[var]["values"][points.hole_ids == hole_id]
                    values = np.array(
                        [self.code_to_cat_map[name][var][val] for val in values]
                    )
                    log.add_categorical_point_data(
                        var, depths, values, cat_to_color_map.get(var, None)
                    )

                for var in continuous_point_vars:
                    pass

            log.create_figure(y_axis_label="Depth (m)", title=hole_id)

            return log.fig

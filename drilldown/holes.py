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


def categorical_color_map(colors):
    mapping = np.linspace(0, len(colors) - 1, 256)
    new_colors = np.empty((256, 3))
    i_pre = -np.inf
    for i, color in enumerate(colors):
        new_colors[(mapping > i_pre) & (mapping <= i)] = list(color)
        i_pre = i
    return ListedColormap(new_colors)


class DrillHole:
    def __init__(self, name, workspace=None):
        self.name = name
        if workspace == None:
            self.workspace = Workspace()
        else:
            self.workspace = workspace

        self.hole_group = DrillholeGroup.create(self.workspace)
        self.vars = []
        self.categorical_vars = []
        self.continuous_vars = []
        self.intervals = {}
        self.categorical_mapping = {}
        self.categorical_color_map = {}

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
            name=self.name,
            parent=self.hole_group,
        )

    def add_from_to(self, from_to):
        if isinstance(from_to, pd.core.frame.DataFrame):
            from_to = from_to.values
        self.from_to = from_to.astype(np.float64)

        return self.from_to

    def make_from_to(self, depths, connected=True):
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

    def add_intervals(
        self, name, data, categorical_color_rng=666, categorical_pastel_factor=0.2
    ):
        if isinstance(data, pd.core.series.Series):
            data = data.values
        data, _type = convert_array_type(data, return_type=True)
        if _type == "str":
            data, unique_values = pd.factorize(data)
            colors = distinctipy.get_colors(
                len(unique_values),
                pastel_factor=categorical_pastel_factor,
                rng=categorical_color_rng,
            )
            self.categorical_mapping[name] = {
                index: {"name": value, "color": color}
                for index, (value, color) in enumerate(zip(unique_values, colors))
            }
            self.categorical_color_map[name] = categorical_color_map(colors)
            self.categorical_vars.append(name)
        else:
            self.continuous_vars.append(name)

        self.vars.append(name)
        self.intervals[name] = {"values": data, "type": _type, "from-to": self.from_to}
        return self.intervals[name]

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
        from_to = self.make_from_to(depths)
        mesh = self._make_line_mesh(from_to[0], from_to[1])

        return mesh

    def make_intervals_mesh(self, name):
        from_depths = self.desurvey(self.from_to[:, 0])
        to_depths = self.desurvey(self.from_to[:, 1])
        intermediate_depths = np.mean([from_depths, to_depths], axis=0)

        mesh = self._make_line_mesh(from_depths, to_depths)

        mesh.cell_data["from"] = self.from_to[:, 0]
        mesh.cell_data["to"] = self.from_to[:, 1]
        mesh.cell_data["hole ID"] = [self.name] * self.from_to.shape[0]
        mesh.cell_data["x"] = intermediate_depths[:, 0]
        mesh.cell_data["y"] = intermediate_depths[:, 1]
        mesh.cell_data["z"] = intermediate_depths[:, 2]
        for var in self.vars:
            data = self.intervals[var]["values"]
            _type = self.intervals[var]["type"]
            # print(_type)
            if _type == "str":
                # print(f"adding {var}")
                mesh[var] = data
            else:
                mesh.cell_data[var] = data

        return mesh

    def make_point_mesh(self, name):
        pass

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
        self, show_collar=True, show_survey=True, name=None, *args, **kwargs
    ):
        intervals_mesh = self.make_intervals_mesh(name)
        p = DrillDownPlotter()
        p.add_intervals(intervals_mesh, *args, **kwargs)

        if show_collar == True:
            collar_mesh = self.make_collar_mesh()
            p.add_collars(collar_mesh)

        if show_survey == True:
            survey_mesh = self.make_survey_mesh()
            p.add_surveys(survey_mesh)

        return p.show()

    def show_points(self, name=None):
        mesh = self.make_survey_mesh(name)

        return mesh.plot()

    def show(self):
        collar_mesh = self.make_collar_mesh()
        survey_mesh = self.make_survey_mesh()
        intervals_mesh = self.make_intervals_mesh(None)

        p = DrillDownPlotter()
        p.add_collars(collar_mesh)
        p.add_surveys(survey_mesh)
        p.add_intervals(intervals_mesh, radius=10)

        return p.show()

    def drill_log(self):
        log = DrillLog()
        depths = self.from_to

        for var in self.categorical_vars:
            values = self.intervals[var]["values"]
            log.add_categorical_interval_data(
                var, depths, values, self.categorical_mapping[var]
            )

        for var in self.continuous_vars:
            values = self.intervals[var]["values"]

            log.add_continuous_interval_data(var, depths, values)

        log.create_figure(y_axis_label="Depth (m)", title=self.name)

        return log.fig


class DrillHoleGroup:
    def __init__(self, name):
        self.name = name
        self._holes = {}
        self.vars = []
        self.categorical_vars = []
        self.continuous_vars = []
        self.intervals = {}
        self.categorical_mapping = {}
        self.categorical_color_map = {}
        self.workspace = Workspace()
        self.hole_ids_with_data = []

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
            hole = DrillHole(hole_id, workspace=self.workspace)

            hole.add_collar(self.collars[self.collars[:, 0] == hole_id, 1:][0])

            surveys = np.hsplit(self.surveys[self.surveys[:, 0] == hole_id, 1:], 3)
            if (surveys[0].shape[0]) > 0:
                hole.add_survey(surveys[0], surveys[1], surveys[2])

                hole._create_hole()

                self._holes[hole_id] = hole

    def add_from_to(self, hole_ids, from_to):
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
                self._holes[id].add_from_to(dataset)

        return self.from_to

    def add_intervals(
        self,
        name,
        hole_ids,
        data,
        categorical_color_rng=666,
        categorical_pastel_factor=0.2,
    ):
        if isinstance(hole_ids, pd.core.series.Series):
            hole_ids = hole_ids.values

        if isinstance(data, pd.core.series.Series):
            data = data.values

        hole_ids = convert_array_type(hole_ids)
        data, _type = convert_array_type(data, return_type=True)
        self.vars.append(name)

        if _type == "str":
            data, unique_values = pd.factorize(data)
            colors = distinctipy.get_colors(
                len(unique_values),
                pastel_factor=categorical_pastel_factor,
                rng=categorical_color_rng,
            )
            self.categorical_mapping[name] = {
                index: {"name": value, "color": color}
                for index, (value, color) in enumerate(zip(unique_values, colors))
            }
            self.categorical_color_map[name] = categorical_color_map(colors)
            self.categorical_vars.append(name)
        else:
            self.continuous_vars.append(name)
        for id in self.hole_ids_with_data:
            if id not in self.intervals.keys():
                self.intervals[id] = {}
            dataset = data[hole_ids == id]
            if dataset.shape[0] > 0:
                self.intervals[id][name] = {
                    "values": dataset,
                    "type": _type,
                    "from-to": self._holes[id].from_to,
                }
        return self.intervals

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
            from_to = hole.make_from_to(depths)
            if from_to.shape[0] > 0:
                if mesh is None:
                    mesh = hole._make_line_mesh(from_to[:, 0], from_to[:, 1])
                else:
                    mesh += hole._make_line_mesh(from_to[:, 0], from_to[:, 1])

        return mesh

    def make_intervals_mesh(self, name):
        meshes = None
        for hole_id in self.hole_ids_with_data:
            hole = self._holes[hole_id]
            from_depths = hole.desurvey(hole.from_to[:, 0])
            to_depths = hole.desurvey(hole.from_to[:, 1])
            intermediate_depths = np.mean([from_depths, to_depths], axis=0)
            if from_depths.shape[0] > 0:
                mesh = hole._make_line_mesh(from_depths, to_depths)
                mesh.cell_data["from"] = hole.from_to[:, 0]
                mesh.cell_data["to"] = hole.from_to[:, 1]
                mesh.cell_data["hole ID"] = [hole_id] * hole.from_to.shape[0]
                mesh.cell_data["x"] = intermediate_depths[:, 0]
                mesh.cell_data["y"] = intermediate_depths[:, 1]
                mesh.cell_data["z"] = intermediate_depths[:, 2]

                for var in self.vars:
                    data = self.intervals[hole_id][var]["values"]
                    _type = self.intervals[hole_id][var]["type"]
                    if _type == "str":
                        mesh[var] = data
                    else:
                        mesh.cell_data[var] = data
                # print(mesh["Stratigraphy"])
                if meshes is None:
                    meshes = mesh
                else:
                    meshes += mesh
        # print(len(meshes["Stratigraphy"]))

        return meshes

    def make_points_mesh(self):
        return

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
        self, show_collars=True, show_surveys=True, name=None, *args, **kwargs
    ):
        intervals_mesh = self.make_intervals_mesh(name)
        p = DrillDownPlotter()
        p.add_intervals(intervals_mesh, *args, **kwargs)

        if show_collars == True:
            collars_mesh = self.make_collars_mesh()
            p.add_collars(collars_mesh)

        if show_surveys == True:
            surveys_mesh = self.make_surveys_mesh()
            p.add_surveys(surveys_mesh)

        return p.show()

    def show(self):
        collars_mesh = self.make_collars_mesh()
        surveys_mesh = self.make_surveys_mesh()
        intervals_mesh = self.make_intervals_mesh(None)

        p = DrillDownPlotter()
        p.add_collars(collars_mesh)
        p.add_surveys(surveys_mesh)
        p.add_intervals(intervals_mesh, radius=10)

        return p.show()

    def drill_log(self, hole_id):
        hole = self._holes[hole_id]
        log = DrillLog()
        depths = hole.from_to
        for var in self.categorical_vars:
            values = self.intervals[hole_id][var]["values"]
            log.add_categorical_interval_data(
                var, depths, values, self.categorical_mapping[var]
            )

        for var in self.continuous_vars:
            values = self.intervals[hole_id][var]["values"]
            log.add_continuous_interval_data(var, depths, values)

        log.create_figure(y_axis_label="Depth (m)", title=hole_id)

        return log.fig

import pyvista as pv
from geoh5py.workspace import Workspace
from geoh5py.groups import DrillholeGroup
from geoh5py.objects import Drillhole
import numpy as np
import pandas as pd

from .plotter import DrillDownPlotter
from .drill_log import DrillLog


class DrillHole:
    def __init__(self, name, workspace=None):
        self.name = name
        if workspace == None:
            self.workspace = Workspace()
        else:
            self.workspace = workspace

        self.hole_group = DrillholeGroup.create(self.workspace)
        self.vars = []

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

    def add_data(self, name, data):
        if isinstance(data, pd.core.series.Series):
            data = data.values

        self.vars.append(name)
        data_added = self._hole.add_data(
            {name: {"values": data.astype(np.float64), "from-to": self.from_to}}
        )
        return data_added

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
        mesh = self._make_line_mesh(from_depths, to_depths)

        mesh.cell_data["from"] = self.from_to[:, 0]
        mesh.cell_data["to"] = self.from_to[:, 1]
        mesh.cell_data["hole ID"] = [self.name] * self.from_to.shape[0]
        for var in self.vars:
            mesh.cell_data[var] = self._hole.get_data(var)[0].values

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
        for var in self.vars:
            values = self._hole.get_data(var)[0].values
            log.add_continuous_interval_data(depths, values, var)

        log.create_figure(y_axis_label="Depth (m)", title=self.name)

        return log.fig


class DrillHoleGroup:
    def __init__(self, name):
        self.name = name
        self._holes = {}
        self.vars = []
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

    def add_data(self, name, hole_ids, data):
        if isinstance(hole_ids, pd.core.series.Series):
            hole_ids = hole_ids.values

        if isinstance(data, pd.core.series.Series):
            data = data.values

        data = np.c_[hole_ids, data]
        self.vars.append(name)

        data_added = {}
        for id in self.hole_ids_with_data:
            dataset = data[:, 1][data[:, 0] == id]
            if dataset.shape[0] > 0:
                data_added[id] = self._holes[id].add_data(
                    name, data[:, 1][data[:, 0] == id]
                )
        return data_added

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
            if from_depths.shape[0] > 0:
                mesh = hole._make_line_mesh(from_depths, to_depths)
                mesh.cell_data["from"] = hole.from_to[:, 0]
                mesh.cell_data["to"] = hole.from_to[:, 1]
                mesh.cell_data["hole ID"] = [hole_id] * hole.from_to.shape[0]
                for var in self.vars:
                    mesh.cell_data[var] = hole._hole.get_data(var)[0].values
                if meshes is None:
                    meshes = mesh
                else:
                    meshes += mesh

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
        for var in self.vars:
            values = hole._hole.get_data(var)[0].values
            log.add_continuous_interval_data(depths, values, var)

        log.create_figure(y_axis_label="Depth (m)", title=hole_id)

        return log.fig

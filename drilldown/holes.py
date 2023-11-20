import pyvista as pv
from geoh5py.workspace import Workspace
from geoh5py.groups import DrillholeGroup
from geoh5py.objects import Drillhole
import numpy as np

# from plotter import DrillDownPlotter


class DrillHoleGroup:
    def __init__(self, name):
        self.name = name
        self._holes = {}
        self.vars = []
        self.workspace = Workspace()

    def add_collars(self, hole_id, collars):
        self.hole_ids = np.unique(hole_id)
        self.collars = np.c_[hole_id, collars]

    def add_surveys(self, hole_id, dist, azm, dip):
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

                hole.create_hole()

                self._holes[hole_id] = hole

    def add_from_to(self, hole_ids, from_to):
        self.from_to = np.c_[hole_ids, from_to]
        hole_ids = [id for id in np.unique(hole_ids) if id in self._holes.keys()]
        for id in hole_ids:
            self._holes[id].add_from_to(self.from_to[self.from_to[:, 0] == id, 1:])

        return self.from_to

    def add_data(self, name, hole_ids, data):
        self.hole_ids_with_data = np.unique(hole_ids)
        data = np.c_[hole_ids, data]
        self.vars.append(name)

        hole_ids = [id for id in np.unique(hole_ids) if id in self._holes.keys()]
        data_added = {}
        for id in hole_ids:
            data_added[id] = self._holes[id].add_data(
                name, data[:, 1][data[:, 0] == id]
            )
        return data_added

    def desurvey(self, hole_id, depths=None):
        return self._holes[hole_id].desurvey(depths=depths)

    def create_polydata(self):
        self.polydata = self._holes[self.hole_ids_with_data[0]].create_polydata()
        for hole_id in self.hole_ids_with_data[1:]:
            self.polydata += self._holes[hole_id].create_polydata()

        return self.polydata

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
        for hole_id in self._holes.keys():
            hole = self._holes[hole_id]
            from_depths = hole.desurvey(hole.from_to[:, 0])
            to_depths = hole.desurvey(hole.from_to[:, 1])
            if from_depths.shape[0] > 0:
                mesh = hole._make_line_mesh(from_depths, to_depths)
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
        self.collar = collar

    def add_survey(self, dist, azm, dip):
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

        for var in self.vars:
            mesh.cell_data[var] = self._hole.get_data(var)[0].values

        return mesh

    def make_point_mesh(self, name):
        pass

    def show_collar(self):
        mesh = self.make_collar_mesh()

        return mesh.plot()

    def show_survey(self):
        mesh = self.make_survey_mesh()

        return mesh.plot()

    def show_intervals(self, name=None):
        mesh = self.make_intervals_mesh(name)

        return mesh.plot()

    def show_points(self, name=None):
        mesh = self.make_survey_mesh(name)

        return mesh.plot()
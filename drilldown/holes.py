import pyvista as pv
from geoh5py.workspace import Workspace
from geoh5py.groups import DrillholeGroup
from geoh5py.objects import Drillhole
import numpy as np

class DrillHoleGroup(): 
    def __init__(self, name): 
        self.name = name
        self._holes = {}
        self.vars = []
        self.workspace=Workspace()

    def add_collars(self, hole_id, collars): 
        self.hole_ids = np.unique(hole_id)
        self.collars = np.c_[hole_id, collars]

    def add_surveys(self, hole_id, dist, azm, dip): 
        self.surveys = np.c_[hole_id, dist, azm, dip]
    
    def create_holes(self): 
        for hole_id in self.hole_ids: 
            hole = DrillHole(hole_id, workspace=self.workspace)

            hole.add_collar(self.collars[self.collars[:, 0] == hole_id, 1:][0])

            surveys = np.hsplit(self.surveys[self.surveys[:, 0] == hole_id, 1:], 3)
            hole.add_survey(surveys[0], surveys[1], surveys[2])

            hole.create_hole()
            
            self._holes[hole_id] = hole

    def add_from_to(self, hole_id, from_to): 
        self.from_to = np.c_[hole_id, from_to]

        for hole_id in np.unique(hole_id): 
            self._holes[hole_id].add_from_to(self.from_to[self.from_to[:, 0] == hole_id, 1:])
            
        return self.from_to
    
    def add_data(self, name, hole_id, data): 
        self.hole_ids_with_data = np.unique(hole_id)
        data = np.c_[hole_id, data]
        self.vars.append(name)
        data_added = {}
        for hole_id in self.hole_ids_with_data: 
            data_added[hole_id] = self._holes[hole_id].add_data(name, data[:,1][data[:, 0] == hole_id])
        return data_added
    
    def desurvey(self, hole_id, depths=None): 
        return self._holes[hole_id].desurvey(depths=depths)
    
    def create_polydata(self): 
        self.polydata = self._holes[self.hole_ids_with_data[0]].create_polydata()
        for hole_id in self.hole_ids_with_data[1:]: 
            self.polydata += self._holes[hole_id].create_polydata()

        return self.polydata


class DrillHole(): 
    def __init__(self, name, workspace=None): 
        self.name = name
        if workspace==None: 
            self.workspace = Workspace()
        else: 
            self.workspace = workspace

        self.hole_group = DrillholeGroup.create(self.workspace)
        self.vars = []
    
    def add_collar(self, collar): 
        self.collar = collar

    def add_survey(self, dist, azm, dip): 
        self.survey = np.c_[dist, azm, dip]

    def create_hole(self): 
        self._hole = Drillhole.create(
            self.workspace, 
            collar=self.collar, 
            surveys=self.survey, 
            name=self.name, 
            parent=self.hole_group
            )

    def add_from_to(self, from_to):
        self.from_to = from_to.astype(np.float64)

        return self.from_to

    def add_data(self, name, data): 
        self.vars.append(name)
        data_added = self._hole.add_data({name: {"values": data.astype(np.float64), "from-to": self.from_to}})
        return data_added

    def desurvey(self, depths=None): 
        if depths==None: 
            return self._hole.desurvey(self.from_to[:,0]), self._hole.desurvey(self.from_to[:,1])
        else: 
            return self._hole.desurvey(depths)

    def create_polydata(self): 
        from_depth, to_depth = self.desurvey()
        depths = np.empty((from_depth.shape[0] + to_depth.shape[0], 3))
        depths[0::2,:] = from_depth
        depths[1::2,:] = to_depth
        n_connected = np.ones(int(depths.shape[0]/2), dtype="int")*2
        from_positions = np.arange(0, depths.shape[0]-1, 2)
        to_positions = np.arange(1, depths.shape[0], 2)
        depth_connectivity = np.hstack(np.stack([n_connected, from_positions, to_positions], axis=1))
        mesh = pv.PolyData(depths, lines=depth_connectivity)
        
        for var in self.vars:
            mesh.cell_data[var] = self._hole.get_data(var)[0].values

        return mesh
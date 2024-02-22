import collections.abc
import pyvista as pv
from geoh5py.workspace import Workspace
from geoh5py.groups import DrillholeGroup
from geoh5py.objects import Drillhole
import numpy as np
import pandas as pd
import distinctipy
from matplotlib.colors import ListedColormap

from .plotter import DrillDownPlotter
from .drill_log import DrillLog
from .utils import (
    convert_to_numpy_array,
    convert_array_type,
    encode_categorical_data,
    make_categorical_cmap,
    make_color_map_fractional,
)


class HoleData:
    def __init__(self):
        self.hole_ids = []
        self.vars_all = []
        self.categorical_vars = []
        self.continuous_vars = []
        self._depths = None
        self.data = {}
        self.image_var_names = []
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
        image_var_names=[],
        return_data=False,
        construct_categorical_cmap=False,
    ):
        # add vars
        self.vars_all += var_names

        # save flag to construct categorical color map
        self.construct_categorical_cmap = construct_categorical_cmap

        # add hole IDs
        hole_ids = convert_to_numpy_array(hole_ids)
        self.hole_ids = hole_ids
        self.unique_hole_ids = np.unique(hole_ids)

        # encode hole IDs, as strings are wiped in pyvista meshes
        hole_ids_encoded, hole_ids_unique = pd.factorize(hole_ids)
        self.cat_to_code_map["hole ID"] = {
            hole_id: code for code, hole_id in enumerate(hole_ids_unique)
        }
        self.code_to_cat_map["hole ID"] = {
            code: hole_id for code, hole_id in enumerate(hole_ids_unique)
        }
        self.categorical_vars.append("hole ID")

        # add from-to depths
        depths = convert_to_numpy_array(depths)
        self.depths = depths

        # add data
        data = convert_to_numpy_array(data, collapse_dim=False)
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

        self.image_var_names = image_var_names

        if return_data == True:
            return self.data

    # def _desurvey(self, hole_id, depths=None):  # NEW
    #     return self._holes[hole_id].desurvey(depths=depths)

    # def desurvey(self, surveys):
    #     if not isinstance(surveys, Surveys):
    #         raise TypeError("Surveys must be a Surveys object.")

    #     self.surveys = surveys
    #     for hole_id in self.hole_ids:
    #         hole = surveys._holes[hole_id]
    #         depths = hole.desurvey()

    def _construct_categorical_cmap(
        self, var_names=[], cycle=True, rng=999, pastel_factor=0.2
    ):
        if len(var_names) == 0:
            var_names = [
                var
                for var in self.categorical_vars
                if var not in self.cat_to_color_map.keys()
            ]

        for var in var_names:
            categories = self.cat_to_code_map[var].keys()
            cat_to_color_map, matplotlib_formatted_color_maps = make_categorical_cmap(
                categories, cycle=cycle, rng=rng, pastel_factor=pastel_factor
            )

            self.cat_to_color_map[var] = cat_to_color_map
            self.matplotlib_formatted_color_maps[var] = matplotlib_formatted_color_maps
            self.code_to_color_map[var] = {
                code: cat_to_color_map[cat]
                for code, cat in self.code_to_cat_map[var].items()
            }
            # codes = self.code_to_cat_map[var].keys()
            # n_colors = len(codes)

            # if cycle == True:
            #     colors = get_cycled_colors(n_colors)

            # elif cycle == False:
            #     colors = distinctipy.get_colors(
            #         n_colors,
            #         pastel_factor=self.categorical_pastel_factor,
            #         rng=self.categorical_color_rng,
            #     )

            # # create categorical color map
            # self.cat_to_color_map[var] = {
            #     cat: color
            #     for cat, color in zip(self.cat_to_code_map[var].keys(), colors)
            # }

            # # create encoded categorical color map
            # self.code_to_color_map[var] = {
            #     code: color for code, color in zip(codes, colors)
            # }

            # # create matplotlib categorical color map
            # self.matplotlib_formatted_color_maps[
            #     var
            # ] = make_matplotlib_categorical_color_map(colors)

    def add_categorical_cmap(self, var_name, cmap=None, cycle=True):
        if var_name not in self.categorical_vars:
            raise ValueError(f"Data for {var_name} not present.")

        if cmap is None:
            self._construct_categorical_cmap([var_name], cycle=cycle)
        else:
            if not isinstance(cmap, dict):
                raise TypeError("Categorical color map must be a dictionary.")

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
            self.matplotlib_formatted_color_maps[var_name] = ListedColormap(
                [self.code_to_color_map[var_name][code] for code in codes]
            )

    @property
    def depths(self):
        return self._depths

    @depths.setter
    def depths(self, depths):
        if depths is not None:
            depths = convert_to_numpy_array(depths)
        self._depths = depths.astype(np.float64)


class Points(HoleData):
    def __init__(self):
        super().__init__()

        self.mesh = None
        self.surveys = None
        self.collars = None

    def add_data(
        self,
        var_names,
        hole_ids,
        depths,
        data,
        return_data=False,
        construct_categorical_cmap=False,
        **kwargs,
    ):
        super().add_data(
            var_names,
            hole_ids,
            depths,
            data,
            return_data=return_data,
            construct_categorical_cmap=construct_categorical_cmap,
            **kwargs,
        )

    def desurvey(self, surveys):
        if not isinstance(surveys, Surveys):
            raise TypeError("Surveys must be a Surveys object.")

        self.surveys = surveys
        self.collars = surveys.collars

        self.depths_desurveyed = np.empty((self.depths.shape[0], 3))

        for id in self.unique_hole_ids:
            hole_filter = self.hole_ids == id
            hole = self.surveys._holes[id]
            depths = self.depths[hole_filter]
            depths_desurveyed = hole.desurvey(depths)
            self.depths_desurveyed[hole_filter] = depths_desurveyed

    def make_mesh(self):
        meshes = None

        for id in self.unique_hole_ids:
            hole_filter = self.hole_ids == id
            depths = self.depths[hole_filter]

            if self.depths.shape[0] > 0:
                depths_desurveyed = self.depths_desurveyed[hole_filter]

                mesh = pv.PolyData(depths_desurveyed)

                for var in self.vars_all:
                    data = self.data[var]["values"][hole_filter]
                    _type = self.data[var]["type"]
                    if _type == "str":
                        mesh[var] = data
                    else:
                        mesh.point_data[var] = data

                mesh.point_data["depth"] = depths
                mesh.point_data["hole ID"] = [
                    self.cat_to_code_map["hole ID"][id]
                ] * depths.shape[0]

                mesh.point_data["x"] = depths_desurveyed[:, 0]
                mesh.point_data["y"] = depths_desurveyed[:, 1]
                mesh.point_data["z"] = depths_desurveyed[:, 2]

                self.continuous_vars += ["depth", "x", "y", "z"]

                if meshes is None:
                    meshes = mesh
                else:
                    meshes += mesh

        self.mesh = meshes

        return meshes

    def show(self, show_collars=False, show_surveys=False, *args, **kwargs):
        if self.mesh is None:
            self._construct_categorical_cmap()

        p = DrillDownPlotter()
        p.matplotlib_formatted_color_maps = self.matplotlib_formatted_color_maps
        p.add_points(self, "points", selectable=False, *args, **kwargs)

        if show_collars == True:
            p.add_collars(self.collars)

        if show_surveys == True:
            p.add_surveys(self.surveys)

        return p.show()

    def drill_log(self, hole_id, log_vars=[]):
        if hole_id not in self.unique_hole_ids:
            raise ValueError(f"Hole ID {hole_id} not present.")

        if self.construct_categorical_cmap == True:
            # ensure that color maps exist for categorical vars
            self._construct_categorical_cmap()

        log = DrillLog()

        depths = self.depths[self.hole_ids == hole_id]

        if isinstance(log_vars, str):
            log_vars = [log_vars]
        if len(log_vars) == 0:
            log_vars = self.categorical_vars + self.continuous_vars

        for var in log_vars:
            if var in self.categorical_vars:
                values = self.data[var]["values"][self.hole_ids == hole_id]
                cat_to_color_map = self.cat_to_color_map.get(var, None)
                log.add_categorical_point_data(var, depths, values, cat_to_color_map)

            elif var in self.continuous_vars:
                values = self.data[var]["values"][self.hole_ids == hole_id]

                log.add_continuous_point_data(var, depths, values)

            else:
                raise ValueError(f"Data for variable {var} not present.")

        log.create_figure(y_axis_label="Depth (m)", title=hole_id)

        return log.fig


class Intervals(HoleData):
    def __init__(self):
        super().__init__()

        self.mesh = None
        self.surveys = None
        self.collars = None

    def add_data(
        self,
        var_names,
        hole_ids,
        depths,
        data,
        return_data=False,
        construct_categorical_cmap=True,
        **kwargs,
    ):
        super().add_data(
            var_names,
            hole_ids,
            depths,
            data,
            return_data=return_data,
            construct_categorical_cmap=construct_categorical_cmap,
            **kwargs,
        )

    def desurvey(self, surveys):
        if not isinstance(surveys, Surveys):
            raise TypeError("Surveys must be a Surveys object.")

        self.surveys = surveys
        self.collars = surveys.collars

        self.from_depths_desurveyed = np.empty((self.depths.shape[0], 3))
        self.to_depths_desurveyed = np.empty((self.depths.shape[0], 3))
        self.intermediate_depths_desurveyed = np.empty((self.depths.shape[0], 3))

        for id in self.unique_hole_ids:
            hole_filter = self.hole_ids == id
            from_to = self.depths[hole_filter]
            hole = self.surveys._holes[id]

            from_depths_desurveyed = hole.desurvey(from_to[:, 0])
            to_depths_desurveyed = hole.desurvey(from_to[:, 1])
            intermediate_depths_desurveyed = np.mean(
                [from_depths_desurveyed, to_depths_desurveyed], axis=0
            )

            self.from_depths_desurveyed[hole_filter] = from_depths_desurveyed
            self.to_depths_desurveyed[hole_filter] = to_depths_desurveyed
            self.intermediate_depths_desurveyed[
                hole_filter
            ] = intermediate_depths_desurveyed

    def make_mesh(self):
        meshes = None

        for id in self.unique_hole_ids:
            hole = self.surveys._holes[id]
            hole_filter = self.hole_ids == id
            from_to = self.depths[hole_filter]

            if from_to.shape[0] > 0:
                from_depths_desurveyed = self.from_depths_desurveyed[hole_filter]
                to_depths_desurveyed = self.to_depths_desurveyed[hole_filter]
                intermediate_depths_desurveyed = self.intermediate_depths_desurveyed[
                    hole_filter
                ]

                mesh = hole._make_line_mesh(
                    from_depths_desurveyed, to_depths_desurveyed
                )

                mesh.cell_data["from"] = from_to[:, 0]
                mesh.cell_data["to"] = from_to[:, 1]
                mesh.cell_data["hole ID"] = [
                    self.cat_to_code_map["hole ID"][id]
                ] * from_to.shape[0]

                mesh.cell_data["x"] = intermediate_depths_desurveyed[:, 0]
                mesh.cell_data["y"] = intermediate_depths_desurveyed[:, 1]
                mesh.cell_data["z"] = intermediate_depths_desurveyed[:, 2]

                self.continuous_vars += ["from", "to", "x", "y", "z"]

                for var in self.vars_all:
                    data = self.data[var]["values"][hole_filter]
                    _type = self.data[var]["type"]
                    if _type == "str":
                        mesh[var] = data
                    else:
                        mesh.cell_data[var] = data
                if meshes is None:
                    meshes = mesh
                else:
                    meshes += mesh

        self.mesh = meshes

        return meshes

    def show(self, show_collars=False, show_surveys=False, *args, **kwargs):
        if self.mesh is None:
            self._construct_categorical_cmap()

        p = DrillDownPlotter()
        p.add_intervals(self, "intervals", selectable=False, *args, **kwargs)

        if show_collars == True:
            p.add_collars(self.collars)

        if show_surveys == True:
            p.add_surveys(self.surveys)

        return p.show()

    def drill_log(self, hole_id, log_vars=[]):
        if hole_id not in self.unique_hole_ids:
            raise ValueError(f"Hole ID {hole_id} not present.")

        if self.construct_categorical_cmap == True:
            # ensure that color maps exist for categorical vars
            self._construct_categorical_cmap()

        log = DrillLog()

        from_to = self.depths[self.hole_ids == hole_id]

        if isinstance(log_vars, str):
            log_vars = [log_vars]
        if len(log_vars) == 0:
            log_vars = self.categorical_vars + self.continuous_vars

        for var in log_vars:
            if var in self.categorical_vars:
                values = self.data[var]["values"][self.hole_ids == hole_id]
                values = np.array([self.code_to_cat_map[var][val] for val in values])
                cat_to_color_map = self.cat_to_color_map.get(var, None)
                log.add_categorical_interval_data(
                    var, from_to, values, cat_to_color_map
                )

            elif var in self.continuous_vars:
                values = self.data[var]["values"][self.hole_ids == hole_id]

                log.add_continuous_interval_data(var, from_to, values)

            else:
                raise ValueError(f"Data for variable {var} not present.")

        log.create_figure(y_axis_label="Depth (m)", title=hole_id)

        return log.fig


class Collars:
    def __init__(self):
        self.unique_hole_ids = None
        self.coords = None
        self.mesh = None

    def add_data(self, hole_ids, coords):
        hole_ids = convert_to_numpy_array(hole_ids)
        coords = convert_to_numpy_array(coords)

        if coords.ndim != 2:
            raise ValueError("Coordinates must be 2-dimensional.")

        if coords.shape[1] != 3:
            raise ValueError("Coordinates must have 3 columns.")

        if hole_ids.shape[0] != coords.shape[0]:
            raise ValueError("Hole IDs and coordinates must have the same length.")

        self.unique_hole_ids = np.unique(hole_ids)
        self.coords = np.c_[hole_ids, coords]

    def make_mesh(self):
        mesh = pv.PolyData(np.asarray(self.coords[:, 1:], dtype="float"))
        mesh["hole ID"] = self.coords[:, 0]
        self.mesh = mesh
        return mesh

    def show(self, *args, **kwargs):
        p = DrillDownPlotter()
        p.add_collars(self, *args, **kwargs)

        return p.show()


class Surveys:
    def __init__(self):
        self.unique_hole_ids = None
        self.measurements = None
        self.collars = None
        self._holes = {}
        self.mesh = None

    def add_data(self, hole_ids, dist, azm, dip):
        hole_ids = convert_to_numpy_array(hole_ids)
        dist = convert_to_numpy_array(dist)
        azm = convert_to_numpy_array(azm)
        dip = convert_to_numpy_array(dip)

        if (dist.ndim != 1) | (azm.ndim != 1) | (dip.ndim != 1):
            raise ValueError("Survey measurements must be 1-dimensional.")

        if not hole_ids.shape[0] == dist.shape[0] == azm.shape[0] == dip.shape[0]:
            raise ValueError(
                "Hole IDs and survey measurements must have the same length."
            )

        self.unique_hole_ids = np.unique(hole_ids)
        self.measurements = np.c_[hole_ids, dist, azm, dip]

        if self.collars is not None:
            self._create_holes()

    def locate(self, collars):
        self.collars = collars
        for hole_id in self.unique_hole_ids:
            hole = DrillHole()

            hole.add_collar(collars.coords[collars.coords[:, 0] == hole_id, 1:][0])

            measurements = np.hsplit(
                self.measurements[self.measurements[:, 0] == hole_id, 1:], 3
            )
            if (measurements[0].shape[0]) > 0:
                hole.add_survey(measurements[0], measurements[1], measurements[2])

                hole._create_hole()

                self._holes[hole_id] = hole

    def make_mesh(self):
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

        self.mesh = mesh

        return mesh

    def show(self, show_collars=False, *args, **kwargs):
        p = DrillDownPlotter()
        p.add_surveys(self, *args, **kwargs)

        if show_collars == True:
            p.add_collars(self.collars)

        return p.show()

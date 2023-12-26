from vtk import (
    vtkDataSetMapper,
    vtkExtractSelection,
    vtkSelectionNode,
    vtkSelection,
    vtkPropPicker,
    vtkCellPicker,
    vtkPointPicker,
    vtkCellLocator,
    vtkPointLocator,
    vtkHardwareSelector,
    vtkDataObject,
    vtkCoordinate,
    vtkMapper,
)
import vtk.util.numpy_support as vtknumpy
import pyvista as pv
from pyvista import Plotter

# from pyvista._vtk import vtkMapper

import numpy as np
import pandas as pd
from IPython.display import IFrame
import panel as pn
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from functools import partial

from pyvista.trame.jupyter import show_trame
from .drill_log import DrillLog


class DrillDownPlotter(Plotter):
    """Plotting object for displaying drillholes and related datasets."""

    def __init__(self, *args, **kwargs):
        """Initialize plotter."""

        super().__init__(*args, **kwargs)

        self.height = 600
        self.set_background("white")
        self.enable_trackball_style()
        vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()

        self.collars = None
        self.surveys = None
        self.intervals = {}
        self.points = {}

        self._meshes = {}
        self.mesh_names = []
        self.interval_actors = {}
        self.point_actors = {}
        self.collar_actors = {}
        self.survey_actor = None
        self.interval_actor_names = []
        self.point_actor_names = []

        self._cmaps = plt.colormaps()

        self.cells_per_interval = {}
        self.n_intervals = {}
        self.n_points = {}

        self.categorical_vars = {}
        self.continuous_vars = {}
        self.all_vars = {}

        self.interval_vars = {}
        self.categorical_interval_vars = []
        self.continuous_interval_vars = []
        self.point_vars = {}
        self.categorical_point_vars = []
        self.continuous_point_vars = []

        self.code_to_hole_id_map = {}
        self.hole_id_to_code_map = {}
        self.code_to_cat_map = {}
        self.cat_to_code_map = {}
        self.code_to_color_map = {}
        self.cat_to_color_map = {}
        self.matplotlib_formatted_color_maps = {}

        self._show_collar_labels = True

        self._visibility = {}
        self._opacity = {}

        self._active_var = {}
        self.prev_active_var = {}
        self._cmap = {}
        self.prev_continuous_cmap = {}
        self._cmap_range = {}

        # filter attributes
        self.filter_opacity = {}
        self._data_filter = None
        self._interval_filter = None
        self._interval_cells_filter = None
        self._point_filter = None

        self.filtered_mesh = None
        self.interval_filter_actor = None
        self.point_filter_actor = None
        self.filter_actor = None
        self.interval_filter_actor_name = None
        self.point_filter_actor_name = None
        self.filter_actor_name = None

        self._filtered_cells = []
        self._filtered_points = []
        self._filtered_intervals = []

        # selection attributes
        self.accelerated_selection = {}
        self.selection_color = {}
        self.selection_mesh = None
        self.interval_selection_actor = None
        self.point_selection_actor = None
        self.selection_actor = None
        self.interval_selection_actor_name = None
        self.point_selection_actor_name = None
        self.selection_actor_name = None

        self._picked_cell = None
        self._picked_point = None
        self._selected_cells = []
        self._selected_points = []
        self._selected_intervals = []

        self.actor_picker = vtkPropPicker()
        self.pickers = {}  # multiple to enable selective hardware acceleration

        # track clicks
        self.track_click_position(side="left", callback=self._make_selection)
        self.track_click_position(side="left", callback=self._reset_data, double=True)

    def add_mesh(
        self,
        mesh,
        name=None,
        opacity=1,
        pickable=False,
        filter_opacity=0.1,
        selection_color="magenta",
        accelerated_selection=False,
        *args,
        **kwargs,
    ):
        """Add any PyVista mesh/VTK dataset that PyVista can wrap to the scene.

        Parameters
        ----------
        mesh : pyvista.DataSet or pyvista.MultiBlock or vtk.vtkAlgorithm
            Any PyVista or VTK mesh is supported. Also, any dataset
            that pyvista.wrap() can handle including NumPy arrays of XYZ points.
            Plotting also supports VTK algorithm objects (vtk.vtkAlgorithm
            and vtk.vtkAlgorithmOutput). When passing an algorithm, the
            rendering pipeline will be connected to the passed algorithm
            to dynamically update the scene.

        name : str, optional
            Name assigned to mesh

        Returns
        -------
        pyvista.plotting.actor.Actor
            Actor of the mesh.

        """

        self._meshes[name] = mesh
        self.mesh_names.append(name)
        self.filter_opacity[name] = filter_opacity
        actor = super(DrillDownPlotter, self).add_mesh(
            mesh,
            name=name,
            pickable=pickable,
            show_scalar_bar=False,
            opacity=1,
            *args,
            **kwargs,
        )

        if pickable == True:
            self._make_selectable(actor, selection_color, accelerated_selection)

        self.opacity[name] = opacity

        return actor

    def add_collars_mesh(
        self,
        mesh,
        name="collars",
        opacity=1,
        show_labels=True,
        selectable=True,
        selection_color="magenta",
        filter_opacity=0.1,
        accelerated_selection=False,
        *args,
        **kwargs,
    ):
        actor = self.add_mesh(
            mesh,
            name,
            opacity=opacity,
            render_points_as_spheres=True,
            point_size=15,
            pickable=selectable,
            filter_opacity=filter_opacity,
            selection_color=selection_color,
            accelerated_selection=accelerated_selection,
            *args,
            **kwargs,
        )
        self.collar_actor = actor

        if show_labels == True:
            label_actor = self.add_point_labels(
                mesh, mesh["hole ID"], shape_opacity=0.5, show_points=False
            )
            self.collar_label_actor = label_actor

            return actor, label_actor

        else:
            return actor

    def add_surveys_mesh(self, mesh, name="surveys", opacity=1, *args, **kwargs):
        actor = self.add_mesh(mesh, name, opacity=opacity, *args, **kwargs)
        self.survey_actor = actor
        return actor

    def _add_hole_data_mesh(
        self,
        mesh,
        name=None,
        opacity=1,
        categorical_vars=[],
        continuous_vars=[],
        selectable=True,
        active_var=None,
        cmap=None,
        cmap_range=None,
        selection_color="magenta",
        filter_opacity=0.1,
        accelerated_selection=False,
        nan_opacity=1,
        *args,
        **kwargs,
    ):
        # self.continuous_vars[name] = continuous_vars
        # self.categorical_vars[name] = categorical_vars
        # self.all_vars[name] = continuous_vars + categorical_vars
        if active_var is None:  # default to first variable
            self._active_var[name] = self.all_vars[name][0]
        else:
            self._active_var[name] = active_var

        if cmap is None:
            cmap = "Blues"
        self.cmap[name] = cmap

        if cmap_range is None:
            print("cmap_range is None")
            if name in self.interval_actor_names + self.point_actor_names:
                print("name in interval_actor_names or point_actor_names")
                if name in self.interval_actor_names:
                    array = mesh.cell_data[self.active_var[name]]
                else:
                    array = mesh.point_data[self.active_var[name]]
                min, max = array.min(), array.max()
                cmap_range = (min, max)

        self._cmap_range[name] = cmap_range

        self.nan_opacity = nan_opacity

        actor = self.add_mesh(
            mesh,
            name,
            opacity=opacity,
            pickable=selectable,
            filter_opacity=filter_opacity,
            selection_color=selection_color,
            accelerated_selection=accelerated_selection,
            *args,
            **kwargs,
        )
        if active_var is None:  # default to first variable
            self.active_var = (name, self.all_vars[name][0])
        else:
            self.active_var = (name, active_var)
        self.cmap = (name, cmap)
        if cmap_range is None:
            self.reset_cmap_range(name)
        else:
            self.cmap_range = (name, cmap_range)

        self.reset_camera()

        return actor

    def add_intervals_mesh(
        self,
        mesh,
        name=None,
        opacity=1,
        categorical_vars=[],
        continuous_vars=[],
        selectable=True,
        radius=1.5,
        n_sides=20,
        capping=False,
        active_var=None,
        cmap=None,
        cmap_range=None,
        selection_color="magenta",
        filter_opacity=0.1,
        accelerated_selection=False,
        nan_opacity=1,
        *args,
        **kwargs,
    ):
        """Add a PyVista mesh/VTK dataset representing drillhole intervals to the scene.

        Parameters
        ----------
        mesh : pyvista.PolyData or vtk.vtkPolyData
            PyVista mesh/VTK dataset representing drillhole intervals.
        selectable : bool, optional
            Make the mesh available for selection. By default True
        radius : float, optional
            Minimum hole radius (minimum because the radius may vary). By default 1.5
        n_sides : int, optional
            Number of sides for the hole. By default 20
        capping : bool, optional
            Enable or disable capping of each hole interval. By default True
        active_var : str, optional
            Variable corresponding to default scalar array used to color hole intervals. By default None.
        cmap : str, optional
            Matplotlib color map used to color interval data. By default "Blues"
        cmap_range : tuple, optional
            Minimum and maximum value between which color map is applied. By default None
        selection_color : ColorLike, optional
            Color used to color the selection object. By default "#000000"
        accelerated_selection : bool, optional
            When True, accelerates selection using two methods:
            1.) adding a vtkCellLocator object to the vtkCellPicker object
            2.) using hardware selection.
            The latter is less accurate than normal picking. Thus, activating accelerated selection
            increases selection speed but decreases selection accuracy. By default False
        """
        self.interval_actor_names.append(name)
        # self.categorical_interval_vars += categorical_vars
        # self.continuous_interval_vars += continuous_vars
        self.n_intervals[name] = mesh.n_lines
        if capping == True:
            self.cells_per_interval[name] = n_sides + 2
        else:
            self.cells_per_interval[name] = n_sides

        mesh = mesh.tube(radius=radius, n_sides=n_sides, capping=capping)
        actor = self._add_hole_data_mesh(
            mesh,
            name,
            opacity=opacity,
            categorical_vars=categorical_vars,
            continuous_vars=continuous_vars,
            selectable=selectable,
            active_var=active_var,
            cmap=cmap,
            cmap_range=cmap_range,
            filter_opacity=filter_opacity,
            selection_color=selection_color,
            accelerated_selection=accelerated_selection,
            nan_opacity=nan_opacity,
            *args,
            **kwargs,
        )
        self.interval_actors[name] = actor

        return actor

    def add_points_mesh(
        self,
        mesh,
        name=None,
        opacity=1,
        categorical_vars=[],
        continuous_vars=[],
        point_size=10,
        selectable=True,
        active_var=None,
        cmap=None,
        cmap_range=None,
        selection_color="magenta",
        filter_opacity=0.1,
        accelerated_selection=False,
        nan_opacity=1,
        *args,
        **kwargs,
    ):
        self.point_actor_names.append(name)
        self.categorical_point_vars += categorical_vars
        self.continuous_point_vars += continuous_vars
        self.n_points[name] = mesh.n_points

        actor = self._add_hole_data_mesh(
            mesh,
            name,
            opacity=opacity,
            categorical_vars=categorical_vars,
            continuous_vars=continuous_vars,
            point_size=point_size,
            render_points_as_spheres=True,
            selectable=selectable,
            active_var=active_var,
            cmap=cmap,
            cmap_range=cmap_range,
            filter_opacity=filter_opacity,
            selection_color=selection_color,
            accelerated_selection=accelerated_selection,
            nan_opacity=nan_opacity,
            *args,
            **kwargs,
        )
        self.point_actors[name] = actor

        return actor

    def add_collars(self, collars, opacity=1):
        self.collars = collars
        name = "collars"
        if collars.mesh is None:
            collars.make_mesh()

        collars_actor = self.add_collars_mesh(collars.mesh, name, opacity=opacity)

        return collars_actor

    def add_surveys(self, surveys, opacity=1):
        self.surveys = surveys
        name = "surveys"
        if surveys.mesh is None:
            surveys.make_mesh()

        surveys_actor = self.add_surveys_mesh(surveys.mesh, name, opacity=opacity)

        return surveys_actor

    def _add_hole_data(self, data, name):
        self.continuous_vars[name] = data.continuous_vars
        self.categorical_vars[name] = data.categorical_vars
        self.all_vars[name] = data.continuous_vars + data.categorical_vars

        self.hole_id_to_code_map[name] = data.hole_id_to_code_map
        self.code_to_hole_id_map[name] = data.code_to_hole_id_map
        self.cat_to_code_map[name] = data.cat_to_code_map
        self.code_to_cat_map[name] = data.code_to_cat_map
        self.code_to_color_map[name] = data.code_to_color_map
        self.cat_to_color_map[name] = data.cat_to_color_map
        self.matplotlib_formatted_color_maps[
            name
        ] = data.matplotlib_formatted_color_maps

    def add_intervals(
        self,
        intervals,
        name,
        opacity=1,
        selectable=True,
        radius=1.5,
        n_sides=20,
        capping=False,
        active_var=None,
        cmap=None,
        cmap_range=None,
        selection_color="magenta",
        filter_opacity=0.1,
        accelerated_selection=False,
        nan_opacity=1,
        *args,
        **kwargs,
    ):
        self.intervals[name] = intervals
        self.interval_vars[name] = intervals.vars_all
        self.categorical_interval_vars += intervals.categorical_vars
        self.continuous_interval_vars += intervals.continuous_vars
        self._add_hole_data(intervals, name)

        if intervals.mesh is None:
            intervals.make_mesh()

        intervals_actor = self.add_intervals_mesh(
            intervals.mesh,
            name,
            opacity=opacity,
            selectable=selectable,
            radius=radius,
            n_sides=n_sides,
            capping=capping,
            active_var=active_var,
            cmap=cmap,
            cmap_range=cmap_range,
            selection_color=selection_color,
            filter_opacity=filter_opacity,
            accelerated_selection=accelerated_selection,
            nan_opacity=nan_opacity,
            *args,
            **kwargs,
        )

        return intervals_actor

    def add_points(
        self,
        points,
        name,
        opacity=1,
        point_size=10,
        selectable=True,
        active_var=None,
        cmap=None,
        cmap_range=None,
        selection_color="magenta",
        filter_opacity=0.1,
        accelerated_selection=False,
        nan_opacity=1,
        *args,
        **kwargs,
    ):
        self.points[name] = points
        self.point_vars[name] = points.vars_all
        self.categorical_point_vars += points.categorical_vars
        self.continuous_point_vars += points.continuous_vars
        self._add_hole_data(points, name)

        if points.mesh is None:
            points.make_mesh()

        points_actor = self.add_points_mesh(
            points.mesh,
            name,
            opacity=opacity,
            point_size=point_size,
            selectable=selectable,
            active_var=active_var,
            cmap=cmap,
            cmap_range=cmap_range,
            selection_color=selection_color,
            filter_opacity=filter_opacity,
            accelerated_selection=accelerated_selection,
            nan_opacity=nan_opacity,
            *args,
            **kwargs,
        )

        return points_actor

    def add_holes(self, holes, name=None, *args, **kwargs):
        # add color-category and code-category maps
        self.code_to_hole_id_map = holes.code_to_hole_id_map
        self.hole_id_to_code_map = holes.hole_id_to_code_map
        self.code_to_cat_map = holes.code_to_cat_map
        self.cat_to_code_map = holes.cat_to_code_map
        self.code_to_color_map = holes.code_to_color_map
        self.cat_to_color_map = holes.cat_to_color_map
        self.matplotlib_formatted_color_maps = holes.matplotlib_formatted_color_maps

        # make and add collars mesh
        collars_mesh = holes.make_collars_mesh()
        self.add_collars_mesh(collars_mesh)

        # make and add surveys mesh
        surveys_mesh = holes.make_surveys_mesh()
        self.add_surveys_mesh(surveys_mesh)

        # make and add intervals mesh(es)
        for name in holes.intervals.keys():
            intervals_mesh = holes.make_intervals_mesh(name)
            self.add_intervals_mesh(
                intervals_mesh,
                name,
                holes.categorical_interval_vars,
                holes.continuous_interval_vars,
            )

        # make and add points mesh(es)
        for name in holes.points.keys():
            points_mesh = holes.make_points_mesh(name)
            self.add_points_mesh(
                points_mesh,
                name,
                holes.categorical_point_vars,
                holes.continuous_point_vars,
            )

    @property
    def cmaps(self):
        return self._cmaps

    def _make_selectable(
        self,
        actor,
        selection_color="magenta",
        accelerated_selection=False,
    ):
        name = actor.name
        self.selection_color[name] = selection_color
        self.accelerated_selection[name] = accelerated_selection

        if (name in self.interval_actor_names) or (
            name == self.interval_filter_actor_name
        ):
            picker = self._make_intervals_selectable(actor, accelerated_selection)
            self.pickers[name] = picker

        elif (name in self.point_actor_names) or (name == self.point_filter_actor_name):
            picker = self._make_points_selectable(actor, accelerated_selection)
            self.pickers[name] = picker

        # elif name == "collars":
        #     picker = self._make_points_selectable(actor, accelerated_selection)
        #     self.pickers[name] = picker

    def _make_intervals_selectable(self, actor, accelerated_selection=False):
        # set cell picker
        cell_picker = vtkCellPicker()
        cell_picker.SetTolerance(0.0005)

        if accelerated_selection == True:
            # add locator for acceleration
            cell_locator = vtkCellLocator()
            cell_locator.SetDataSet(actor.mapper.dataset)
            cell_locator.BuildLocator()
            cell_picker.AddLocator(cell_locator)

            # use hardware selection for acceleration
            hw_selector = vtkHardwareSelector()
            hw_selector.SetFieldAssociation(vtkDataObject.FIELD_ASSOCIATION_CELLS)
            hw_selector.SetRenderer(self.renderer)

        return cell_picker

    def _make_points_selectable(self, actor, accelerated_selection=False):
        # set point picker
        point_picker = vtkPointPicker()
        point_picker.SetTolerance(0.005)

        if accelerated_selection == True:
            # add locator for acceleration
            point_locator = vtkPointLocator()
            point_locator.SetDataSet(actor.mapper.dataset)
            point_locator.BuildLocator()
            point_picker.AddLocator(point_locator)

            # use hardware selection for acceleration
            hw_selector = vtkHardwareSelector()
            hw_selector.SetFieldAssociation(vtkDataObject.FIELD_ASSOCIATION_POINTS)
            hw_selector.SetRenderer(self.renderer)

        return point_picker

    def _make_selection(self, *args):
        pos = self.click_position + (0,)
        actor_picker = self.actor_picker
        actor_picker.Pick(pos[0], pos[1], pos[2], self.renderer)
        picked_actor = actor_picker.GetActor()
        if picked_actor is not None:
            name = picked_actor.name
            if name == self.filter_actor_name:
                selection_on_filter = True
                name = name.split(" ")[0]

            else:
                selection_on_filter = False

            if name in self.interval_actor_names:
                self._reset_point_selection()
                self._make_intervals_selection(
                    name, pos, selection_on_filter=selection_on_filter
                )
                self._update_data_selection_object(
                    name, selection_on_filter=selection_on_filter
                )

            elif name in self.point_actor_names:
                self._reset_interval_selection()
                self._make_points_selection(
                    name, pos, selection_on_filter=selection_on_filter
                )
                self._update_data_selection_object(
                    name, selection_on_filter=selection_on_filter
                )

            # elif name == "collars":
            #     self._make_collars_selection(pos)
            #     self._update_collar_selection_object(name)

    # def _make_collars_selection(self, pos):
    #     self.actors["intervals"].SetPickable(False)
    #     point_picker = self.pickers["collars"]
    #     point_picker.Pick(pos[0], pos[1], pos[2], self.renderer)
    #     picked_point = point_picker.GetPointId()
    #     if picked_point is not None:
    #         if picked_point == -1:
    #             return

    #         self._make_single_point_selection("collars", picked_point)
    #         self._picked_point = picked_point
    #     self.actors["intervals"].SetPickable(True)

    def _make_intervals_selection(self, name, pos, selection_on_filter=False):
        if selection_on_filter == True:
            cell_picker = self.pickers[name + " filter"]
        else:
            cell_picker = self.pickers[name]

        cell_picker.Pick(pos[0], pos[1], pos[2], self.renderer)
        picked_cell = cell_picker.GetCellId()
        # if name == self.interval_filter_actor_name:
        #     name = name.split(" ")[0]

        if picked_cell is not None:
            # if cell_picker.GetActor().name in self.point_actor_names:
            #     return

            shift_pressed = self.iren.interactor.GetShiftKey()
            ctrl_pressed = self.iren.interactor.GetControlKey()
            if shift_pressed == True:
                self._make_continuous_multi_interval_selection(
                    name, picked_cell, selection_on_filter=selection_on_filter
                )

            elif ctrl_pressed == True:
                self._make_discontinuous_multi_interval_selection(
                    name, picked_cell, selection_on_filter=selection_on_filter
                )

            else:
                self._make_single_interval_selection(
                    name, picked_cell, selection_on_filter=selection_on_filter
                )

            if selection_on_filter == True:
                picked_cell = self._filtered_cells[picked_cell]

            self._picked_cell = picked_cell

    def _make_points_selection(self, name, pos, selection_on_filter=False):
        if selection_on_filter == True:
            point_picker = self.pickers[name + " filter"]
        else:
            point_picker = self.pickers[name]

        point_picker.Pick(pos[0], pos[1], pos[2], self.renderer)
        picked_point = point_picker.GetPointId()
        if picked_point is not None:
            if picked_point == -1:
                # or (
                #     point_picker.GetActor().name in self.interval_actor_names
                # ):
                return
            # if name == self.point_filter_actor_name:
            #     name = name.split(" ")[0]
            shift_pressed = self.iren.interactor.GetShiftKey()
            ctrl_pressed = self.iren.interactor.GetControlKey()
            if shift_pressed == True:
                self._make_continuous_multi_point_selection(
                    name, picked_point, selection_on_filter=selection_on_filter
                )

            elif ctrl_pressed == True:
                self._make_discontinuous_multi_point_selection(
                    name, picked_point, selection_on_filter=selection_on_filter
                )

            else:
                self._make_single_point_selection(
                    name, picked_point, selection_on_filter=selection_on_filter
                )

            if selection_on_filter == True:
                picked_point = self._filtered_points[picked_point]

            self._picked_point = picked_point

    def _make_single_interval_selection(
        self, name, picked_cell, selection_on_filter=False
    ):
        cells_per_interval = self.cells_per_interval[name]
        selected_interval = int(np.floor(picked_cell / cells_per_interval))
        selected_cells = np.arange(
            selected_interval * cells_per_interval,
            (selected_interval + 1) * cells_per_interval,
        ).tolist()

        if selection_on_filter == True:
            selected_interval = self._filtered_intervals[selected_interval]
            selected_cells = list(self._filtered_cells[selected_cells])

        self._selected_intervals = [selected_interval]
        self._selected_cells = selected_cells

    def _make_single_point_selection(
        self, name, picked_point, selection_on_filter=False
    ):
        if selection_on_filter == True:
            picked_point = self._filtered_points[picked_point]

        self._selected_points = [picked_point]

    def _make_discontinuous_multi_interval_selection(
        self, name, picked_cell, selection_on_filter=False
    ):
        cells_per_interval = self.cells_per_interval[name]
        selected_interval = int(np.floor(picked_cell / cells_per_interval))
        selected_cells = np.arange(
            selected_interval * cells_per_interval,
            (selected_interval + 1) * cells_per_interval,
        ).tolist()

        if selection_on_filter == True:
            selected_interval = self._filtered_intervals[selected_interval]
            selected_cells = list(self._filtered_cells[selected_cells])

        self._selected_intervals += [selected_interval]
        self._selected_cells += selected_cells

    def _make_discontinuous_multi_point_selection(
        self, name, picked_point, selection_on_filter=False
    ):
        pass

    def _make_continuous_multi_interval_selection(
        self, name, picked_cell, selection_on_filter=False
    ):
        cells_per_interval = self.cells_per_interval[name]
        if selection_on_filter == True:
            prev_picked_cell = np.where(self._filtered_cells == self._picked_cell)[0][0]
            prev_selected_intervals = np.where(
                np.isin(self._filtered_intervals, self._selected_intervals[:-1])
            )[0].tolist()
            prev_selected_intervals += np.where(
                np.isin(self._filtered_intervals, self._selected_intervals[-1])
            )[
                0
            ].tolist()  #  needed as np.isin or np.where seems to sort the output and the resulting first interval should be last

        else:
            prev_picked_cell = self._picked_cell
            prev_selected_intervals = self._selected_intervals

        if prev_picked_cell is not None:
            if prev_picked_cell < picked_cell:  # normal direction (down the hole)
                selected_intervals = np.arange(
                    prev_selected_intervals[-1] + 1,
                    int(np.floor(picked_cell / cells_per_interval)) + 1,
                ).tolist()
                selected_cells = np.arange(
                    (selected_intervals[0]) * cells_per_interval,
                    (selected_intervals[-1] + 1) * cells_per_interval,
                ).tolist()

                if selection_on_filter == True:
                    selected_intervals = list(
                        self._filtered_intervals[selected_intervals]
                    )
                    selected_cells = list(self._filtered_cells[selected_cells])

                self._selected_intervals += selected_intervals
                self._selected_cells += selected_cells

            else:  # reverse direction (up the hole)
                selected_intervals = np.arange(
                    int(np.floor(picked_cell / cells_per_interval)),
                    prev_selected_intervals[-1],
                ).tolist()
                selected_cells = np.arange(
                    (selected_intervals[0] * cells_per_interval),
                    (selected_intervals[-1] + 1) * cells_per_interval,
                ).tolist()

                if selection_on_filter == True:
                    selected_intervals = list(
                        self._filtered_intervals[selected_intervals]
                    )
                    selected_cells = list(self._filtered_cells[selected_cells])

                self._selected_intervals = selected_intervals + self._selected_intervals
                self._selected_cells = selected_cells + self._selected_cells

    def _make_continuous_multi_point_selection(
        self, name, picked_point, selection_on_filter
    ):
        pass  # not trivial as cell IDs are not inherently sequential along hole

    # def _reset_collar_selection(self):
    #     self._picked_point = None
    #     self._selected_points = []
    #     self.remove_actor(self.selection_actor)
    #     self.selection_actor = None
    #     self.selection_actor_name = None
    #     self.point_selection_actor = None

    def _reset_interval_selection(self):
        self._picked_cell = None
        self._selected_intervals = []
        self._selected_cells = []

        self.remove_actor(self.selection_actor)
        self.selection_actor = None
        self.selection_mesh = None
        self.selection_actor_name = None
        self.interval_selection_actor = None
        self.interval_selection_actor_name = None

    def _reset_point_selection(self):
        self._picked_point = None
        self._selected_points = []

        self.remove_actor(self.selection_actor)
        self.selection_actor = None
        self.selection_mesh = None
        self.selection_actor_name = None
        self.point_selection_actor = None
        self.point_selection_actor_name = None

    def _reset_data_selection(self, *args):
        pos = self.click_position + (0,)
        actor_picker = self.actor_picker
        actor_picker.Pick(pos[0], pos[1], pos[2], self.renderer)
        picked_actor = actor_picker.GetActor()
        if picked_actor is not None:
            name = picked_actor.name
            if name in self.interval_actor_names:
                self._reset_point_selection()
            elif name in self.point_actor_names:
                self._reset_interval_selection()
            else:
                return
        else:
            self._reset_interval_selection()
            self._reset_point_selection()
        # self._reset_collar_selection()

    def _reset_interval_filter(self):
        self._picked_cell = None
        self._filtered_intervals = []
        self._filtered_cells = []

        for name in self.interval_actor_names + self.point_actor_names:
            self.actors[name].prop.opacity = 1

        if self.interval_filter_actor is not None:
            self.remove_actor(self.interval_filter_actor)
            self.actors[
                self.interval_filter_actor_name.replace(" filter", "")
            ].SetPickable(True)

        self.filter_actor = None
        self.filter_actor_name = None
        self.interval_filter_actor = None
        self.interval_filter_actor_name = None

    def _reset_point_filter(self):
        self._picked_point = None
        self._filtered_points = []

        for name in self.interval_actor_names + self.point_actor_names:
            self.actors[name].prop.opacity = 1

        if self.point_filter_actor is not None:
            self.remove_actor(self.selection_actor)
            self.actors[
                self.point_filter_actor_name.replace(" filter", "")
            ].SetPickable(True)

        self.filter_actor = None
        self.filter_actor_name = None
        self.point_filter_actor = None
        self.interval_filter_actor_name = None

    def _reset_data_filter(self, *args):
        pos = self.click_position + (0,)
        actor_picker = self.actor_picker
        actor_picker.Pick(pos[0], pos[1], pos[2], self.renderer)
        picked_actor = actor_picker.GetActor()
        if picked_actor is not None:
            name = picked_actor.name
            if name in self.interval_actor_names:
                self._reset_point_filter()
            elif name in self.point_actor_names:
                self._reset_interval_filter()
            else:
                return
        else:
            self._reset_interval_filter()
            self._reset_point_filter()
        # self._reset_collar_filter

    def _reset_data(self, *args):
        if self.selection_actor is not None:
            self._reset_data_selection()
            return
        else:
            self._reset_data_filter()
            return

    # def
    # def _update_collar_filter_object(self, name):
    #     filtered_name = "collars filter"
    #     mesh = self._meshes[name]
    #     filtered_points = self._filtered_points
    #     filtered_mesh = mesh.extract_points(filtered_points)
    #     if (filtered_mesh.n_points!= 0) and (filtered_mesh.n_cells != 0):
    #         self.filtered_mesh = filtered_mesh
    #         filter_actor = self.add_mesh(
    #             filtered_mesh,
    #             filtered_name,
    #             point_size=20,
    #             render_points_as_spheres=True,
    #             reset_camera=False,
    #             pickable=True,
    #         )
    #     self.filter_actor = filter_actor
    #     self.filtered_hole_ids = filtered_mesh["hole ID"]
    #     return filter_actor

    def _update_data_filter_object(self, name):
        filtered_name = name + " filter"
        self.filter_actor_name = filtered_name

        if name in self.interval_actor_names:
            self.interval_filter_actor_name = filtered_name
            filter_actor = self._update_interval_filter_object(name)
            self.interval_filter_actor = filter_actor

        elif name in self.point_actor_names:
            self.point_filter_actor_name = filtered_name
            filter_actor = self._update_point_filter_object(name)
            self.point_filter_actor = filter_actor

        self.filter_actor = filter_actor

        if filter_actor is not None:
            filter_actor.mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
                0, -5
            )

        # update non-selected
        self.actors[name].prop.opacity = self.filter_opacity[name]
        self.actors[name].SetPickable(False)

        # update selected holes
        self._selected_hole_ids = [
            self.code_to_hole_id_map[name][id]
            for id in np.unique(self.filter_mesh["hole ID"])
        ]
        self.render()
        pass

    def _update_interval_filter_object(self, name):
        filtered_name = name + " filter"

        # update selected
        mesh = self._meshes[name]
        filtered_cells = self._filtered_cells
        filter_mesh = mesh.extract_cells(filtered_cells)
        if (filter_mesh.n_points != 0) and (filter_mesh.n_cells != 0):
            self.filter_mesh = filter_mesh

            filter_actor = self.add_mesh(
                filter_mesh,
                filtered_name,
                scalars=self.active_var[name],
                cmap=self.cmap[name],
                clim=self.cmap_range[name],
                reset_camera=False,
                pickable=True,
                selection_color=self.selection_color[name],
                accelerated_selection=self.accelerated_selection[name],
            )
            return filter_actor

    def _update_point_filter_object(self, name):
        filtered_name = name + " filter"

        # update selected
        mesh = self._meshes[name]
        filtered_points = self._filtered_points
        filter_mesh = mesh.extract_points(filtered_points)
        if (filter_mesh.n_points != 0) and (filter_mesh.n_cells != 0):
            self.filter_mesh = filter_mesh
            filter_actor = self.add_mesh(
                filter_mesh,
                filtered_name,
                point_size=10,
                render_points_as_spheres=True,
                scalars=self.active_var[name],
                cmap=self.cmap[name],
                clim=self.cmap_range[name],
                reset_camera=False,
                pickable=True,
                selection_color=self.selection_color[name],
                accelerated_selection=self.accelerated_selection[name],
            )
            return filter_actor

    # def _update_collar_selection_object(self, name):
    #     selection_name = "collars selection"

    #     mesh = self._meshes[name]
    #     selected_points = self._selected_points
    #     selection_mesh = mesh.extract_points(selected_points)
    #     if (selection_mesh.n_points != 0) and (selection_mesh.n_cells != 0):
    #         self.selection_mesh = selection_mesh
    #         selection_actor = self.add_mesh(
    #             selection_mesh,
    #             selection_name,
    #             point_size=20,
    #             render_points_as_spheres=True,
    #             color=self.selection_color[name],
    #             reset_camera=False,
    #             pickable=False,
    #         )
    #         self.selection_actor = selection_actor

    #         self.selected_hole_ids = selection_mesh["hole ID"]
    #         return selection_actor

    def _update_data_selection_object(self, name, selection_on_filter=False):
        selection_name = name + " selection"
        self.selection_actor_name = selection_name

        if name in self.interval_actor_names:
            self.interval_selection_actor_name = selection_name
            selection_actor = self._update_interval_selection_object(
                name, selection_on_filter=selection_on_filter
            )
            self.interval_selection_actor = selection_actor

        elif name in self.point_actor_names:
            self.point_selection_actor_name = selection_name
            selection_actor = self._update_point_selection_object(
                name, selection_on_filter=selection_on_filter
            )
            self.point_selection_actor = selection_actor

        else:
            return

        self.selection_actor = selection_actor

        if selection_actor is not None:
            selection_actor.mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
                0, -6
            )

            # update selected holes
            self._selected_hole_ids = [
                self.code_to_hole_id_map[name][id]
                for id in np.unique(self.selection_mesh["hole ID"])
            ]
        self.render()

    def _update_interval_selection_object(self, name, selection_on_filter=False):
        selection_name = name + " selection"

        # update selected
        mesh = self._meshes[name]
        selected_cells = self._selected_cells
        selection_mesh = mesh.extract_cells(selected_cells)
        if (selection_mesh.n_points != 0) and (selection_mesh.n_cells != 0):
            self.selection_mesh = selection_mesh

            selection_actor = self.add_mesh(
                selection_mesh,
                selection_name,
                color=self.selection_color[name],
                reset_camera=False,
                pickable=False,
            )
            return selection_actor

    def _update_point_selection_object(self, name, selection_on_filter=False):
        selection_name = name + " selection"

        # update selected
        mesh = self._meshes[name]
        selected_points = self._selected_points
        selection_mesh = mesh.extract_points(selected_points)
        if (selection_mesh.n_points != 0) and (selection_mesh.n_cells != 0):
            self.selection_mesh = selection_mesh
            selection_actor = self.add_mesh(
                selection_mesh,
                selection_name,
                point_size=10,
                render_points_as_spheres=True,
                color=self.selection_color[name],
                reset_camera=False,
                pickable=False,
            )
            return selection_actor

    @property
    def active_var(self):
        return self._active_var

    @active_var.setter
    def active_var(self, key_value_pair):
        name, active_var = key_value_pair
        self.prev_active_var[name] = active_var
        self._active_var[name] = active_var

        actor = self.actors.get(name, None)
        if actor is not None:
            actor.mapper.dataset.set_active_scalars(active_var)

            if (self.selection_actor is not None) and (
                self.selection_actor_name == name + " selection"
            ):
                self.selection_actor.mapper.dataset.set_active_scalars(active_var)

        if active_var in self.categorical_vars[name]:
            cmap = self.matplotlib_formatted_color_maps.get(active_var, None)
            self.cmap = (name, cmap)
            cmap_range = list(self.code_to_cat_map[name][active_var].keys())[-1]
            self.cmap_range = (
                name,
                (0, cmap_range),
            )
        elif active_var in self.continuous_vars[name]:
            if self.prev_active_var[name] in self.categorical_vars[name]:
                cmap = self.prev_continuous_cmap[name]
                self.cmap = (name, cmap)

            self.reset_cmap_range(name)

        self.render()

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, key_value_pair):
        name, cmap = key_value_pair
        self._cmap[name] = cmap

        actors = []
        actor = self.actors.get(name, None)
        actors.append(actor)
        if actor is not None:
            if (self.selection_actor is not None) and (
                self.selection_actor_name == name + " selection"
            ):
                actors.append(self.selection_actor)
            for actor in actors:
                print("updating cmap")
                if self.active_var[name] in self.continuous_vars[name]:
                    actor.mapper.lookup_table.cmap = cmap
                    self.prev_continuous_cmap[name] = cmap

                else:
                    actor.mapper.lookup_table = pv.LookupTable(cmap)
                actor.mapper.lookup_table.nan_color = "white"  # self.nan_opacity

        self.render()

    @property
    def cmap_range(self):
        return self._cmap_range

    @cmap_range.setter
    def cmap_range(self, key_value_pair):
        name, cmap_range = key_value_pair
        self._cmap_range[name] = cmap_range

        actors = []
        actor = self.actors.get(name, None)
        if actor is not None:
            actors.append(actor)
            if (self.selection_actor is not None) and (
                self.selection_actor_name == name + " selection"
            ):
                actors.append(self.selection_actor)

            for actor in actors:
                actor.mapper.lookup_table.scalar_range = cmap_range
                actor.mapper.SetUseLookupTableScalarRange(True)

        self.render()

    def reset_cmap_range(self, name=None):
        if name is None:
            names = self._mesh.keys()
        else:
            names = [name]
        for name in names:
            mesh = self._meshes[name]
            if name in self.interval_actor_names:
                array = mesh.cell_data[self.active_var[name]]
            elif name in self.point_actor_names:
                array = mesh.point_data[self.active_var[name]]
            min, max = array.min(), array.max()
            self.cmap_range = (name, (min, max))

    @property
    def visibility(self):
        return self._visibility

    @visibility.setter
    def visibility(self, key_value_pair):
        name, visible = key_value_pair
        self._visibility[name] = visible

        actor = self.actors.get(name, None)
        if actor is not None:
            if visible == True:
                self.actors[name].prop.opacity = 1
            else:
                self.actors[name].prop.opacity = 0

        self.render()

    @property
    def opacity(self):
        return self._opacity

    @opacity.setter
    def opacity(self, key_value_pair):
        name, opacity = key_value_pair
        self._opacity[name] = opacity
        actor = self.actors.get(name, None)
        if actor is not None:
            self.actors[name].prop.opacity = opacity

        self.render()

    @property
    def selected_data(self):
        return self._selected_data

    @property
    def selected_intervals(self):
        return self._selected_intervals

    @selected_intervals.setter
    def selected_intervals(self, key_value_pair):
        name, intervals = key_value_pair
        interval_cells = []
        for interval in intervals:
            interval_cells += np.arange(
                interval * self.cells_per_interval,
                (interval + 1) * self.cells_per_interval,
            ).tolist()

        self._selected_intervals = intervals
        self._selected_cells = interval_cells

        self._update_data_selection_object(name)

    @property
    def filtered_holes(self):
        return self._filtered_holes

    # @filtered_holes.setter
    # def filtered_holes(self, hole_ids):

    @property
    def data_filter(self):
        return self._data_filter

    @data_filter.setter
    def data_filter(self, filter_input):
        if isinstance(filter_input, tuple):
            name, filter = filter_input
        else:
            filter = filter_input
            data_names = self.interval_actor_names + self.point_actor_names
            if len(data_names) != 1:
                raise ValueError(
                    "Multiple datasets are present. Please specify name of dataset to filter."
                )
            name = data_names[0]

        if name in self.interval_actor_names:
            self.interval_filter = (name, filter)
        elif name in self.point_actor_names:
            self.point_filter = (name, filter)

    @property
    def interval_filter(self):
        return self._interval_filter

    @interval_filter.setter
    def interval_filter(self, filter_input):
        if isinstance(filter_input, tuple):
            name, filter = filter_input
        else:
            filter = filter_input
            if len(self.interval_actor_names) != 1:
                raise ValueError(
                    "Multiple interval datasets are present. Please specify name of dataset to filter."
                )
            name = self.interval_actor_names[0]

        if len(filter) != self.n_intervals[name]:  # filter entire dataset
            raise ValueError(
                "Filter must be of length equal to number of intervals in dataset."
            )
        self._interval_filter = np.array(filter)
        self._interval_cells_filter = np.repeat(filter, self.cells_per_interval[name])

        self._filtered_intervals = np.arange(self.n_intervals[name])[
            self.interval_filter
        ]
        self._filtered_cells = np.arange(
            self.n_intervals[name] * self.cells_per_interval[name]
        )[self._interval_cells_filter]

        self._update_data_filter_object(name)

    @property
    def point_filter(self):
        return self._point_filter

    @point_filter.setter
    def point_filter(self, filter_input):
        if isinstance(filter_input, tuple):
            name, filter = filter_input
        else:
            filter = filter_input
            if len(self.point_actor_names) != 1:
                raise ValueError(
                    "Multiple point datasets are present. Please specify name of dataset to filter."
                )
            name = self.point_actor_names[0]

        if len(filter) != self.n_points[name]:  # filter entire dataset
            raise ValueError(
                "Filter must be of length equal to number of points in dataset."
            )

        self._point_filter = np.array(filter)
        self._filtered_points = np.arange(self.n_points[name])[self.point_filter]

        self._update_data_filter_object(name)

    def convert_selection_to_filter(self):
        if self.interval_selection_actor_name is not None:
            self.convert_interval_selection_to_filter()
        elif self.point_selection_actor_name is not None:
            self.convert_point_selection_to_filter()

    def convert_filter_to_selection(self, keep_filter=False):
        if self.interval_filter_actor_name is not None:
            self.convert_interval_filter_to_selection(keep_filter)
        elif self.point_filter_actor_name is not None:
            self.convert_point_filter_to_selection(keep_filter)

    def convert_interval_selection_to_filter(self):
        name = self.interval_selection_actor_name.replace(" selection", "")
        n_intervals = self.n_intervals[name]
        filter = np.isin(np.arange(n_intervals), self._selected_intervals)
        self._reset_interval_selection()
        self.interval_filter = (name, filter)

    def convert_interval_filter_to_selection(self, keep_filter=False):
        self._selected_cells = self._filtered_cells
        self._selected_intervals = self._filtered_intervals
        name = self.interval_filter_actor_name.replace(" filter", "")
        if keep_filter == False:
            self._reset_interval_filter()
        self._update_data_selection_object(name)

    def convert_point_selection_to_filter(self):
        name = self.point_selection_actor_name.replace(" selection", "")
        n_points = self.n_points[name]
        filter = np.isin(np.arange(n_points), self._selected_points)
        self._reset_point_selection()
        self.point_filter = (name, filter)

    def convert_point_filter_to_selection(self, keep_filter=False):
        self._selected_points = self._filtered_points
        name = self.point_filter_actor_name.replace(" filter", "")
        if keep_filter == False:
            self._reset_point_filter()
        self._update_data_selection_object(name)

    @property
    def show_collar_labels(self):
        return self._collar_labels_visible

    @show_collar_labels.setter
    def show_collar_labels(self, visible):
        self._collar_labels_visible = visible
        if visible == True:
            self.add_actor(
                self.collar_label_actor, name="collar labels", reset_camera=False
            )
        elif visible == False:
            self.remove_actor(self.collar_label_actor)

    def _process_data_output(self, name, indices, step=1):
        holes_mesh = self._meshes[name]
        exclude_vars = [
            "TubeNormals",
            "vtkOriginalPointIds",
            "vtkOriginalCellIds",
        ]  # added by pyvista
        vars = [var for var in holes_mesh.array_names if var not in exclude_vars]
        data_dict = {}
        for var in vars:
            data_dict[var] = holes_mesh[var][::step][indices]

        data = pd.DataFrame(data_dict)
        data["hole ID"] = [
            self.code_to_hole_id_map[name][code] for code in data["hole ID"]
        ]
        for var in self.categorical_vars[name]:
            data[var] = data[var].astype("category")
            data[var] = [self.code_to_cat_map[name][var][code] for code in data[var]]

        return data

    def _process_selected_data_output(self, indices, step=1):
        selection_name = self.selection_actor_name
        name = selection_name.replace(" selection", "")
        data = self._process_data_output(name, indices, step)

        return data

    def selected_interval_data(self):
        intervals = self._selected_intervals
        selection_name = self.selection_actor_name
        name = selection_name.replace(" selection", "")
        data = self._process_selected_data_output(
            intervals, self.cells_per_interval[name]
        )

        return data

    def selected_point_data(self):
        points = self._selected_points
        data = self._process_selected_data_output(points)

        return data

    def selected_data(self):
        name = self.selection_actor_name.replace(" selection", "")
        if name in self.interval_actor_names:
            return self.selected_interval_data()
        elif name in self.point_actor_names:
            return self.selected_point_data()

    def all_interval_data(self, name=None):
        if name is None:
            names = self.interval_actor_names
        else:
            names = [name]

        all_data = {}
        for name in names:
            intervals = np.arange(self.n_intervals[name])
            data = self._process_data_output(
                name, intervals, self.cells_per_interval[name]
            )
            all_data[name] = data

        if len(names) == 1:
            return all_data[name]
        else:
            return all_data

    def all_point_data(self, name=None):
        if name is None:
            names = self.point_actor_names
        else:
            names = [name]

        all_data = {}
        for name in names:
            points = np.arange(self.n_points[name])
            data = self._process_data_output(name, points)
            all_data[name] = data

        if len(names) == 1:
            return all_data[name]
        else:
            return all_data

    def all_data(self, name=None):
        if name is not None:
            if name in self.interval_actor_names:
                return self.all_interval_data(name)
            elif name in self.point_actor_names:
                return self.all_point_data(name)
        else:
            all_point_data = self.all_point_data()
            all_interval_data = self.all_interval_data()

            return all_point_data, all_interval_data

    @property
    def selected_hole_ids(self):
        return self._selected_hole_ids

    @selected_hole_ids.setter
    def selected_hole_ids(self, hole_ids):
        if isinstance(hole_ids, str):
            hole_ids = [hole_ids]
        self._selected_hole_ids = hole_ids

        # filter interval data
        for name in self.interval_actor_names:
            interval_data = self.all_interval_data(name)
            interval_filter = interval_data["hole ID"].isin(hole_ids)
            self.interval_filter = (name, interval_filter)

        # filter point data
        for name in self.point_actor_names:
            point_data = self.all_point_data(name)
            point_filter = point_data["hole ID"].isin(hole_ids)
            self.point_filter = (name, point_filter)

    def selected_drill_log(
        self,
        log_vars=[],
    ):
        interval_data = self.selected_interval_data()
        # point_data = self.selected_point_data()

        hole_id = interval_data["hole ID"].unique()
        if len(hole_id) == 1:
            # check if no variables are passed; if so, use all variables
            if len(log_vars) == 0:
                interval_vars = (
                    self.categorical_interval_vars + self.continuous_interval_vars
                )
                point_vars = self.categorical_point_vars + self.continuous_point_vars
                log_vars = interval_vars + point_vars

            log = DrillLog()

            # add interval data
            depths = interval_data[["from", "to"]].values

            for var in log_vars:
                for name in self.interval_actor_names:
                    if var in self.categorical_vars[name]:
                        cat_to_color_map = self.cat_to_color_map[name]
                        values = interval_data[var].values
                        log.add_categorical_interval_data(
                            var,
                            depths,
                            values,
                            cat_to_color_map.get(var, None),
                        )

                        exit_flag = True
                        break

                    if var in self.continuous_vars[name]:
                        values = interval_data[var].values
                        log.add_continuous_interval_data(var, depths, values)

                        exit_flag = True
                        break

                    if exit_flag == True:
                        break

                # for name in self.point_actor_names:
                #     if var in self.categorical_vars[name]:
                #         cat_to_color_map = self.cat_to_color_map[name]
                #         values = point_data[var].values
                #         log.add_categorical_point_data(
                #             var, depths, values, cat_to_color_map.get(var, None)
                #         )
                #         exit_flag = True
                #         break

                #     if var in self.continuous_vars[name]:
                #         values = point_data[var].values
                #         log.add_continuous_point_data(var, depths, values)
                #         exit_flag = True
                #         break

                #     if exit_flag == True:
                #         break

            log.create_figure(y_axis_label="Depth (m)", title=hole_id[0])

            return log.fig

    def iframe(self, w="100%", h=None):
        if h is None:
            h = self.height

        self._pv_viewer = show_trame(self, mode="server")
        self._trame_viewer = self._pv_viewer.viewer
        self._src = self._pv_viewer.src
        self._server = self._trame_viewer.server
        self._state = self._server.state
        self._iframe = IFrame(self._src, w, h)

        return self._iframe


class DrillDownPanelPlotter(DrillDownPlotter, pn.Row):
    """Plotting object for displaying drillholes and related datasets with simple GUI."""

    def __init__(self, *args, **kwargs):
        """Initialize plotter."""
        self.ctrl_widget_width = 300
        self.active_var_widgets = {}
        self.cmap_widgets = {}
        self.cmap_range_widgets = {}
        self.show_widgets = {}
        self.opacity_widgets = {}

        # create control panel
        self.ctrls = pn.Accordion(
            ("mesh visibility", self._make_mesh_visibility_card()),
            toggle=True,
            width=self.ctrl_widget_width,
            height=self.height,
        )

        super(DrillDownPanelPlotter, self).__init__(*args, **kwargs)

        super(pn.Row, self).__init__(
            self.ctrls, self.iframe(sizing_mode="stretch_both"), height=self.height
        )

    def add_mesh(self, mesh, name=None, add_show_widgets=True, *args, **kwargs):
        """Add any PyVista mesh/VTK dataset that PyVista can wrap to the scene and corresponding widgets to the GUI.

        Parameters
        ----------
        mesh : pyvista.DataSet or pyvista.MultiBlock or vtk.vtkAlgorithm
            Any PyVista or VTK mesh is supported. Also, any dataset
            that pyvista.wrap() can handle including NumPy arrays of XYZ points.
            Plotting also supports VTK algorithm objects (vtk.vtkAlgorithm
            and vtk.vtkAlgorithmOutput). When passing an algorithm, the
            rendering pipeline will be connected to the passed algorithm
            to dynamically update the scene.

        name : str, optional
            Name assigned to mesh

        Returns
        -------
        pyvista.plotting.actor.Actor
            Actor of the mesh.

        """
        actor = super(DrillDownPanelPlotter, self).add_mesh(
            mesh, name=name, *args, **kwargs
        )

        if name == self.selection_actor_name:
            add_show_widgets = False

        if add_show_widgets == True:
            # set up widget to show and hide mesh
            show_widget = pn.widgets.Checkbox(value=True)
            self.show_widgets[name] = show_widget
            # self.show_widgets.append(show_widget)
            show_widget.param.watch(partial(self._on_mesh_show_change, name), "value")

            # set up widget to control mesh opacity
            opacity_widget = pn.widgets.FloatSlider(
                start=0,
                end=1,
                step=0.01,
                value=1,
                show_value=False,
                width=int(2 * self.ctrl_widget_width / 3),
            )
            self.opacity_widgets[name] = opacity_widget
            opacity_widget.param.watch(
                partial(self._on_mesh_opacity_change, name), "value"
            )

            # situate show and opacity widgets; add to mesh visibility widget
            self.mesh_visibility_card.append(pn.pane.Markdown(f"{name}"))
            self.mesh_visibility_card.append(pn.Row(show_widget, opacity_widget))
            self.mesh_visibility_card.append(pn.layout.Divider())

        return actor

    def _add_hole_data_mesh(
        self,
        mesh,
        name=None,
        categorical_vars=[],
        continuous_vars=[],
        selectable=True,
        active_var=None,
        cmap="Blues",
        cmap_range=None,
        selection_color="magenta",
        accelerated_selection=False,
        nan_opacity=1,
        *args,
        **kwargs,
    ):
        actor = super(DrillDownPanelPlotter, self)._add_hole_data_mesh(
            mesh,
            name=name,
            categorical_vars=categorical_vars,
            continuous_vars=continuous_vars,
            selectable=selectable,
            active_var=active_var,
            cmap=cmap,
            cmap_range=cmap_range,
            selection_color=selection_color,
            accelerated_selection=accelerated_selection,
            nan_opacity=nan_opacity,
            *args,
            **kwargs,
        )

        self._make_hole_ctrl_card(name, active_var, cmap, cmap_range)

        return actor

    def add_intervals_mesh(
        self,
        mesh,
        name=None,
        categorical_vars=[],
        continuous_vars=[],
        active_var=None,
        cmap="Blues",
        cmap_range=None,
        *args,
        **kwargs,
    ):
        """Add a PyVista mesh/VTK dataset representing drillhole intervals to the scene. Add corresponding widgets to GUI.

        Parameters
        ----------
        mesh : pyvista.PolyData or vtk.vtkPolyData
            PyVista mesh/VTK dataset representing drillhole intervals.
        active_var : str, optional
            Variable corresponding to default scalar array used to color hole intervals. By default None.
        cmap : str, optional
            Matplotlib color map used to color interval data. By default "Blues"
        cmap_range : tuple, optional
            Minimum and maximum value between which color map is applied. By default None
        """

        super(DrillDownPanelPlotter, self).add_intervals_mesh(
            mesh,
            name=name,
            active_var=active_var,
            categorical_vars=categorical_vars,
            continuous_vars=continuous_vars,
            cmap=cmap,
            cmap_range=cmap_range,
            *args,
            **kwargs,
        )
        # set up widget to show and hide mesh
        if active_var is not None:
            self.active_var = (name, active_var)
        self.cmap = (name, cmap)
        if cmap_range != None:
            self.cmap_range = (name, cmap_range)

    def add_points_mesh(
        self,
        mesh,
        name=None,
        categorical_vars=[],
        continuous_vars=[],
        point_size=10,
        selectable=True,
        active_var=None,
        cmap="Blues",
        cmap_range=None,
        selection_color="magenta",
        accelerated_selection=False,
        nan_opacity=1,
        *args,
        **kwargs,
    ):
        super(DrillDownPanelPlotter, self).add_points_mesh(
            mesh,
            name=name,
            categorical_vars=categorical_vars,
            continuous_vars=continuous_vars,
            point_size=point_size,
            selectable=selectable,
            active_var=active_var,
            cmap=cmap,
            cmap_range=cmap_range,
            selection_color=selection_color,
            accelerated_selection=accelerated_selection,
            nan_opacity=nan_opacity,
            *args,
            **kwargs,
        )
        # set up widget to show and hide mesh
        if active_var is not None:
            self._active_var[name] = active_var
        self._cmap[name] = cmap
        if cmap_range != None:
            self._cmap_range[name] = cmap_range

    def _make_active_var_widget(self, name, active_var=None):
        options = self.all_vars[name]
        if active_var is None:
            active_var = options[0]
        widget = pn.widgets.Select(
            name=f"{name} active variable",
            options=options,
            value=active_var,
            width=int(0.9 * self.ctrl_widget_width),
        )
        widget.param.watch(partial(self._on_active_var_change, name), "value")

        self.active_var_widgets[name] = widget
        return widget

    def _make_cmap_widget(self, name, cmap=None):
        widget = pn.widgets.Select(
            name=f"{name} color map",
            options=self.cmaps,
            value=cmap,
            width=int(0.9 * self.ctrl_widget_width),
        )
        widget.param.watch(partial(self._on_cmap_change, name), "value")

        self.cmap_widgets[name] = widget
        return widget

    def _make_cmap_range_widget(self, name, cmap_range=None):
        min = self.actors[name].mapper.dataset.active_scalars.min()
        max = self.actors[name].mapper.dataset.active_scalars.max()
        if cmap_range == None:
            cmap_range = (min, max)
        widget = pn.widgets.RangeSlider(
            name=f"{name} color map range",
            start=min,
            end=max,
            step=min - max / 1000,
            value=cmap_range,
            width=int(0.9 * self.ctrl_widget_width),
        )
        widget.param.watch(partial(self._on_cmap_range_change, name), "value")
        widget.param.default = cmap_range

        self.cmap_range_widgets[name] = widget
        return widget

    def _make_hole_ctrl_card(
        self, name, active_var=None, cmap="Blues", cmap_range=None
    ):
        self.hole_ctrl_card = pn.Column(
            width=self.ctrl_widget_width,
        )
        self.hole_ctrl_card.append(self._make_active_var_widget(name, active_var))
        self.hole_ctrl_card.append(self._make_cmap_widget(name, cmap))
        self.hole_ctrl_card.append(self._make_cmap_range_widget(name, cmap_range))

        self.ctrls.append((f"{name} controls", self.hole_ctrl_card))
        return self.hole_ctrl_card

    def _make_mesh_visibility_card(self):
        self.mesh_visibility_card = pn.Column(scroll=True, width=self.ctrl_widget_width)

        return self.mesh_visibility_card

    @property
    def active_var(self):
        return self._active_var

    @active_var.setter
    def active_var(self, key_value_pair):
        super(DrillDownPanelPlotter, DrillDownPanelPlotter).active_var.fset(
            self, key_value_pair
        )
        name, active_var = key_value_pair
        if name in self.active_var_widgets.keys():
            self.active_var_widgets[name].value = active_var

            if active_var in self.continuous_vars[name]:
                self.cmap_widgets[name].visible = True
                self.cmap_range_widgets[name].visible = True

            elif active_var in self.categorical_vars[name]:
                self.cmap_widgets[name].visible = False
                self.cmap_range_widgets[name].visible = False

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, key_value_pair):
        super(DrillDownPanelPlotter, DrillDownPanelPlotter).cmap.fset(
            self, key_value_pair
        )
        name, cmap = key_value_pair
        if (name in self.cmap_widgets.keys()) and (
            name in self.cmap_range_widgets.keys()
        ):
            if isinstance(cmap, str):
                self.cmap_widgets[name].value = cmap

    @property
    def cmap_range(self):
        return self._cmap_range

    @cmap_range.setter
    def cmap_range(self, key_value_pair):
        super(DrillDownPanelPlotter, DrillDownPanelPlotter).cmap_range.fset(
            self, key_value_pair
        )
        name, cmap_range = key_value_pair
        if name in self.cmap_range_widgets.keys():
            self.cmap_range_widgets[name].value = cmap_range
            self.cmap_range_widgets[name].step = (cmap_range[1] - cmap_range[0]) / 1000

    @property
    def visibility(self):
        return self._visibility

    @visibility.setter
    def visibility(self, key_value_pair):
        super(DrillDownPanelPlotter, DrillDownPanelPlotter).visibility.fset(
            self, key_value_pair
        )
        name, visible = key_value_pair
        self.show_widgets[name].value = visible

    @property
    def opacity(self):
        return self._opacity

    @opacity.setter
    def opacity(self, key_value_pair):
        super(DrillDownPanelPlotter, DrillDownPanelPlotter).opacity.fset(
            self, key_value_pair
        )
        name, opacity = key_value_pair
        self.opacity_widgets[name].value = opacity

    def _on_active_var_change(self, name, event):
        active_var = event.new
        self.active_var = (name, active_var)

        active_scalars = self.actors[name].mapper.dataset.active_scalars
        self.cmap_range_widgets[name].start = active_scalars.min()
        self.cmap_range_widgets[name].end = active_scalars.max()

    def _on_cmap_change(self, name, event):
        cmap = event.new
        self.cmap = (name, cmap)

        active_scalars = self.actors[name].mapper.dataset.active_scalars
        self.cmap_range_widgets[name].start = active_scalars.min()
        self.cmap_range_widgets[name].end = active_scalars.max()

    def _on_cmap_range_change(self, name, event):
        cmap_range = event.new
        self.cmap_range = (name, cmap_range)

    def _on_mesh_show_change(self, name, event):
        visible = event.new
        self.visibility = (name, visible)

    def _on_mesh_opacity_change(self, name, event):
        opacity = event.new
        self.opacity = (name, opacity)

    def iframe(self, sizing_mode="fixed", w="100%", h=None):
        _iframe = super(DrillDownPanelPlotter, self).iframe()
        _src = _iframe.src
        if h is None:
            h = self.height
        if sizing_mode == "stretch_width":
            w = "100%"
        elif sizing_mode == "stretch_height":
            h = "100%"
        elif sizing_mode == "stretch_both":
            w = "100%"
            h = "100%"

        html = f"""<iframe frameborder="0" title="panel app" style="width: {w};height: {h};flex-grow: 1" src="{_src}"></iframe>"""
        _iframe = pn.pane.HTML(html, sizing_mode=sizing_mode)
        self._iframe = _iframe

        return pn.Column(
            pn.panel(_iframe, sizing_mode=sizing_mode), sizing_mode=sizing_mode
        )

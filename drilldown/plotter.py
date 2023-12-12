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
        self.translate_by = None
        self.set_background("white")
        self.enable_trackball_style()
        vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()

        self._meshes = {}
        self.interval_actors = {}
        self.point_actors = {}
        self.interval_actor_names = []
        self.point_actor_names = []

        self._cmaps = plt.colormaps()

        self.cells_per_interval = {}
        self.n_intervals = {}
        self.n_points = {}

        self.categorical_vars = {}
        self.continuous_vars = {}
        self.all_vars = {}

        self.code_to_hole_id_map = None
        self.hole_id_to_code_map = None
        self.code_to_cat_map = None
        self.cat_to_code_map = None
        self.code_to_color_map = None
        self.cat_to_color_map = None
        self.matplotlib_formatted_color_maps = None

        self._visibility = {}
        self._opacity = {}

        self._active_var = {}
        self.prev_active_var = {}
        self._cmap = {}
        self.prev_continuous_cmap = {}
        self._cmap_range = {}

        # filter attributes
        self._data_filter = None
        self._interval_filter = None
        self._interval_cells_filter = None
        self._point_filter = None

        # selection attributes
        self.selection_color = {}
        self.selection_mesh = None
        self.interval_selection_actor = None
        self.point_selection_actor = None
        self.selection_actor = None
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
        self.track_click_position(
            side="left", callback=self._reset_selection, double=True
        )

    def add_mesh(self, name, mesh, pickable=False, *args, **kwargs):
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
        if self.translate_by is None:
            self.translate_by = [-1 * val for val in mesh.center]
        mesh = mesh.translate(self.translate_by)

        actor = super(DrillDownPlotter, self).add_mesh(
            mesh, name=name, pickable=pickable, show_scalar_bar=False, *args, **kwargs
        )

        return actor

    def add_collars(self, mesh, *args, **kwargs):
        name = "collars"
        actor = self.add_mesh(
            name,
            mesh,
            render_points_as_spheres=True,
            point_size=10,
            *args,
            **kwargs,
        )

        return actor

    def add_surveys(self, mesh, *args, **kwargs):
        name = "surveys"
        actor = self.add_mesh(name, mesh, *args, **kwargs)

        return actor

    def add_hole_data(
        self,
        name,
        mesh,
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
        self.continuous_vars[name] = continuous_vars
        self.categorical_vars[name] = categorical_vars
        self.all_vars[name] = continuous_vars + categorical_vars

        self.nan_opacity = nan_opacity

        actor = self.add_mesh(name, mesh, pickable=selectable, *args, **kwargs)
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

    def add_intervals(
        self,
        name,
        mesh,
        categorical_vars=[],
        continuous_vars=[],
        selectable=True,
        radius=1.5,
        n_sides=20,
        capping=False,
        active_var=None,
        cmap="Blues",
        cmap_range=None,
        selection_color="magenta",
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
        self.n_intervals[name] = mesh.n_lines
        if capping == True:
            self.cells_per_interval[name] = n_sides + 2
        else:
            self.cells_per_interval[name] = n_sides

        mesh = mesh.tube(radius=radius, n_sides=n_sides, capping=capping)
        actor = self.add_hole_data(
            name,
            mesh,
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
        self.interval_actors[name] = actor
        if selectable == True:
            self._make_selectable(actor, selection_color, accelerated_selection)

        return actor

    def add_points(
        self,
        name,
        mesh,
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
        self.point_actor_names.append(name)
        self.n_points[name] = mesh.n_points

        actor = self.add_hole_data(
            name,
            mesh,
            categorical_vars=categorical_vars,
            continuous_vars=continuous_vars,
            point_size=point_size,
            render_points_as_spheres=True,
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
        self.point_actors[name] = actor
        if selectable == True:
            self._make_selectable(actor, selection_color, accelerated_selection)

        return actor

    def add_holes(self, holes, *args, **kwargs):
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
        self.add_collars(collars_mesh)

        # make and add surveys mesh
        surveys_mesh = holes.make_surveys_mesh()
        self.add_surveys(surveys_mesh)

        # make and add intervals mesh(es)
        for name in holes.intervals.keys():
            intervals_mesh = holes.make_intervals_mesh(name)
            self.add_intervals(
                name,
                intervals_mesh,
                holes.categorical_interval_vars,
                holes.continuous_interval_vars,
            )

        # make and add points mesh(es)
        for name in holes.points.keys():
            points_mesh = holes.make_points_mesh(name)
            self.add_points(
                name,
                points_mesh,
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

        if name in self.interval_actor_names:
            picker = self._make_intervals_selectable(actor, accelerated_selection)
            self.pickers[name] = picker

        elif name in self.point_actor_names:
            picker = self._make_points_selectable(actor, accelerated_selection)
            self.pickers[name] = picker

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
        # set cell picker
        point_picker = vtkPointPicker()
        point_picker.SetTolerance(0.0005)

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
            if name in self.interval_actor_names:
                self._make_intervals_selection(name, pos)
                self._update_selection_object(name)

            if name in self.point_actor_names:
                self._make_points_selection(name, pos)
                self._update_selection_object(name)

    def _make_intervals_selection(self, name, pos):
        cell_picker = self.pickers[name]
        cell_picker.Pick(pos[0], pos[1], pos[2], self.renderer)
        picked_cell = cell_picker.GetCellId()

        if picked_cell is not None:
            if cell_picker.GetActor().name in self.point_actor_names:
                return

            shift_pressed = self.iren.interactor.GetShiftKey()
            ctrl_pressed = self.iren.interactor.GetControlKey()
            if shift_pressed == True:
                self._make_continuous_multi_interval_selection(name, picked_cell)

            elif ctrl_pressed == True:
                self._make_discontinuous_multi_interval_selection(name, picked_cell)

            else:
                self._make_single_interval_selection(name, picked_cell)

            self._picked_cell = picked_cell

    def _make_points_selection(self, name, pos):
        point_picker = self.pickers[name]
        point_picker.Pick(pos[0], pos[1], pos[2], self.renderer)
        picked_point = point_picker.GetPointId()
        if picked_point is not None:
            if (picked_point == -1) or (
                point_picker.GetActor().name in self.interval_actor_names
            ):
                return

            shift_pressed = self.iren.interactor.GetShiftKey()
            ctrl_pressed = self.iren.interactor.GetControlKey()
            if shift_pressed == True:
                self._make_continuous_multi_point_selection(name, picked_point)

            elif ctrl_pressed == True:
                self._make_discontinuous_multi_point_selection(name, picked_point)

            else:
                self._make_single_point_selection(name, picked_point)

            self._picked_point = picked_point

    def _make_single_interval_selection(self, name, picked_cell):
        cells_per_interval = self.cells_per_interval[name]
        selected_interval = int(np.floor(picked_cell / cells_per_interval))
        selected_cells = np.arange(
            selected_interval * cells_per_interval,
            (selected_interval + 1) * cells_per_interval,
        ).tolist()

        self._selected_intervals = [selected_interval]
        self._selected_cells = selected_cells

    def _make_single_point_selection(self, name, picked_point):
        self._selected_points = [picked_point]

    def _make_discontinuous_multi_interval_selection(self, name, picked_cell):
        cells_per_interval = self.cells_per_interval[name]
        selected_interval = int(np.floor(picked_cell / cells_per_interval))
        selected_cells = np.arange(
            selected_interval * cells_per_interval,
            (selected_interval + 1) * cells_per_interval,
        ).tolist()

        self._selected_intervals += [selected_interval]
        self._selected_cells += selected_cells

    def _make_discontinuous_multi_point_selection(self, name, picked_point):
        pass

    def _make_continuous_multi_interval_selection(self, name, picked_cell):
        cells_per_interval = self.cells_per_interval[name]
        if self._picked_cell is not None:
            if self._picked_cell < picked_cell:  # normal direction (down the hole)
                selected_intervals = np.arange(
                    self._selected_intervals[-1] + 1,
                    int(np.floor(picked_cell / cells_per_interval)) + 1,
                ).tolist()
                selected_cells = np.arange(
                    (selected_intervals[0]) * cells_per_interval,
                    (selected_intervals[-1] + 1) * cells_per_interval,
                ).tolist()

                self._selected_intervals += selected_intervals
                self._selected_cells += selected_cells

            else:  # reverse direction (up the hole)
                selected_intervals = np.arange(
                    int(np.floor(picked_cell / cells_per_interval)),
                    self._selected_intervals[-1],
                ).tolist()
                selected_cells = np.arange(
                    (selected_intervals[0] * cells_per_interval),
                    (selected_intervals[-1] + 1) * cells_per_interval,
                ).tolist()

                self._selected_intervals = selected_intervals + self._selected_intervals
                self._selected_cells = selected_cells + self._selected_cells

    def _make_continuous_multi_point_selection(self, name, picked_point):
        pass  # not trivial as cell IDs are not inherently sequential along hole

    def _reset_interval_selection(self):
        self._picked_cell = None
        self._selected_intervals = []
        self._selected_cells = []

        for name in self.interval_actor_names + self.point_actor_names:
            self.actors[name].prop.opacity = 1

        self.remove_actor(self.selection_actor)
        self.selection_actor = None
        self.selection_actor_name = None
        self.interval_selection_actor = None

    def _reset_point_selection(self):
        self._picked_point = None
        self._selected_points = []

        for name in self.interval_actor_names + self.point_actor_names:
            self.actors[name].prop.opacity = 1

        self.remove_actor(self.selection_actor)
        self.selection_actor = None
        self.selection_actor_name = None
        self.point_selection_actor = None

    def _reset_selection(self, *args):
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

    def _update_selection_object(self, name):
        selection_name = name + " selection"
        self.selection_actor_name = selection_name

        if name in self.interval_actor_names:
            selection_actor = self._update_interval_selection_object(name)
            self.interval_selection_actor = selection_actor

        elif name in self.point_actor_names:
            selection_actor = self._update_point_selection_object(name)
            self.point_selection_actor = selection_actor

        self.selection_actor = selection_actor

        selection_actor.mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
            0, -5
        )

        # update non-selected
        self.actors[name].prop.opacity = 0.1

        # update selected holes
        self._selected_hole_ids = [
            self.code_to_hole_id_map[name][id]
            for id in np.unique(self.selection_mesh["hole ID"])
        ]
        self.render()

    def _update_interval_selection_object(self, name):
        selection_name = name + " selection"

        # update selected
        mesh = self._meshes[name]
        selected_cells = self._selected_cells
        selection_mesh = mesh.extract_cells(selected_cells)
        self.selection_mesh = selection_mesh

        selection_actor = self.add_mesh(
            selection_name,
            selection_mesh,
            scalars=self.active_var[name],
            cmap=self.cmap[name],
            clim=self.cmap_range[name],
            reset_camera=False,
            pickable=False,
        )
        return selection_actor

    def _update_point_selection_object(self, name):
        selection_name = name + " selection"

        # update selected
        mesh = self._meshes[name]
        selected_points = self._selected_points
        selection_mesh = mesh.extract_points(selected_points)
        self.selection_mesh = selection_mesh
        selection_actor = self.add_mesh(
            selection_name,
            selection_mesh,
            point_size=10,
            render_points_as_spheres=True,
            scalars=self.active_var[name],
            cmap=self.cmap[name],
            clim=self.cmap_range[name],
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

        actor = self.actors[name]
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
        actor = self.actors[name]
        actors.append(actor)
        if (self.selection_actor is not None) and (
            self.selection_actor_name == name + " selection"
        ):
            actors.append(self.selection_actor)
        for actor in actors:
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
        actor = self.actors[name]
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
            names = self.actors.keys()
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

        self._update_selection_object(name)

    @property
    def data_filter(self):
        return self._data_filter

    @data_filter.setter
    def data_filter(self, key_value_pair):
        name, filter = key_value_pair
        if name in self.interval_actor_names:
            self.interval_filter = key_value_pair
        elif name in self.point_actor_names:
            self.point_filter = key_value_pair

    @property
    def interval_filter(self):
        return self._interval_filter

    @interval_filter.setter
    def interval_filter(self, key_value_pair):
        name, filter = key_value_pair
        self._interval_filter = np.array(filter)
        self._interval_cells_filter = np.repeat(filter, self.cells_per_interval[name])

        if len(filter) == len(self._selected_intervals):  # filter only selection
            self._selected_intervals = self._selected_intervals[self._interval_filter]
            self._selected_cells = self._selected_cells[self._interval_cells_filter]
            self._update_selection_object(name)

        elif len(filter) == self.n_intervals[name]:  # filter entire dataset
            self._selected_intervals = np.arange(self.n_intervals[name])[
                self.interval_filter
            ]
            self._selected_cells = np.arange(
                self.n_intervals[name] * self.cells_per_interval[name]
            )[self._interval_cells_filter]
            self._update_selection_object(name)

    @property
    def point_filter(self):
        return self._point_filter

    @point_filter.setter
    def point_filter(self, key_value_pair):
        name, filter = key_value_pair
        self._point_filter = np.array(filter)

        if len(filter) == len(self._selected_points):  # filter only selection
            self._selected_points = self._selected_points[self._point_filter]
            self._update_selection_object(name)

        elif len(filter) == self.n_points[name]:  # filter entire dataset
            self._selected_points = np.arange(self.n_points[name])[self.point_filter]
            self._update_selection_object(name)

    def process_data_output(self, name, indices, step=1):
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

    def process_selected_data_output(self, indices, step=1):
        selection_name = self.selection_actor_name
        name = selection_name.replace(" selection", "")
        data = self.process_data_output(name, indices, step)

        return data

    def selected_interval_data(self):
        intervals = self._selected_intervals
        selection_name = self.selection_actor_name
        name = selection_name.replace(" selection", "")
        data = self.process_selected_data_output(
            intervals, self.cells_per_interval[name]
        )

        return data

    def selected_point_data(self):
        points = self._selected_points
        data = self.process_selected_data_output(points)

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
            data = self.process_data_output(
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
            data = self.process_data_output(name, points)
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
        categorical_interval_vars=[],
        continuous_interval_vars=[],
        categorical_point_vars=[],
        continuous_point_vars=[],
    ):
        interval_data = self.selected_interval_data()
        # point_data = self.selected_point_data()

        hole_id = interval_data["hole ID"].unique()
        if len(hole_id) == 1:
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

            log = DrillLog()

            # add interval data
            depths = interval_data[["from", "to"]].values

            for var in categorical_interval_vars:
                for name in self.interval_actor_names:
                    cat_to_color_map = self.cat_to_color_map[name]
                    if var in cat_to_color_map.keys():
                        values = interval_data[var].values
                        log.add_categorical_interval_data(
                            var,
                            depths,
                            values,
                            cat_to_color_map.get(var, None),
                        )
                        break

            for var in continuous_interval_vars:
                values = interval_data[var].values
                log.add_continuous_interval_data(var, depths, values)

            # add point data

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

    def add_mesh(self, name, mesh, add_show_widgets=True, *args, **kwargs):
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
        actor = super(DrillDownPanelPlotter, self).add_mesh(name, mesh, *args, **kwargs)

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

    def add_hole_data(
        self,
        name,
        mesh,
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
        actor = super(DrillDownPanelPlotter, self).add_hole_data(
            name,
            mesh,
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

    def add_intervals(
        self,
        name,
        mesh,
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

        super(DrillDownPanelPlotter, self).add_intervals(
            name,
            mesh,
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

    def add_points(
        self,
        name,
        mesh,
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
        super(DrillDownPanelPlotter, self).add_points(
            name,
            mesh,
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

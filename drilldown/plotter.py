from vtk import (
    vtkDataSetMapper,
    vtkExtractSelection,
    vtkSelectionNode,
    vtkSelection,
    vtkPropPicker,
    vtkCellPicker,
    vtkCellLocator,
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

        self._actors = {}
        self._filters = {}
        self.selection_extracts = {}
        vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
        # vtkMapper.SetResolveCoincidentTopologyPolygonOffsetParameters(0, -0.5)

    def add_mesh(self, mesh, name, *args, **kwargs):
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
        if self.translate_by is None:
            self.translate_by = [-1 * val for val in mesh.center]
        mesh = mesh.translate(self.translate_by)

        actor = super(DrillDownPlotter, self).add_mesh(mesh, name=name, *args, **kwargs)
        self._actors[name] = actor
        # self.reset_camera()

        return actor

    def add_collars(self, mesh, *args, **kwargs):
        name = "collars"
        self.add_mesh(
            mesh,
            name,
            render_points_as_spheres=True,
            point_size=10,
            pickable=False,
            *args,
            **kwargs,
        )

    def add_surveys(self, mesh, *args, **kwargs):
        name = "surveys"
        self.add_mesh(mesh, name, pickable=False, *args, **kwargs)

    def add_intervals(
        self,
        mesh,
        selectable=True,
        radius=1.5,
        n_sides=20,
        capping=True,
        active_var=None,
        cmap="blues",
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
            Matplotlib color map used to color interval data. By default "blues"
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
        name = "drillhole intervals"
        self.n_sides = n_sides

        if capping == True:
            self._faces_per_interval = self.n_sides + 2
        else:
            self._faces_per_interval = self.n_sides

        filter = mesh.tube(radius=radius, n_sides=self.n_sides, capping=capping)
        self._filters[name] = filter

        self._hole_vars = []
        for var in filter.array_names:
            array = filter[var]
            if array.IsNumeric():
                self._hole_vars.append(var)

        self.nan_opacity = nan_opacity
        self._cmaps = plt.colormaps()
        actor = self.add_mesh(
            filter,
            name=name,
            scalars=active_var,
            show_scalar_bar=False,
            *args,
            **kwargs,
        )
        self.reset_camera()

        self._active_var = active_var
        self._cmap = cmap
        self._cmap_range = cmap_range

        self.active_var = active_var
        self.cmap = cmap
        if self._active_var in self.continuous_vars:
            self.continuous_map = cmap
        if cmap_range != None:
            self.cmap_range = cmap_range

        if selectable == True:
            self._make_selectable(
                actor,
                selection_color=selection_color,
                accelerated_selection=accelerated_selection,
            )

    def add_holes(self, holes, *args, **kwargs):
        # make and add collars mesh
        collars_mesh = holes.make_collars_mesh()
        self.add_collars(collars_mesh)

        # make and add surveys mesh
        surveys_mesh = holes.make_surveys_mesh()
        self.add_surveys(surveys_mesh)

        # make and add intervals mesh
        self.code_to_cat_map = holes.code_to_cat_map
        self.cat_to_code_map = holes.cat_to_code_map
        self.code_to_color_map = holes.code_to_color_map
        self.cat_to_color_map = holes.cat_to_color_map
        self.matplotlib_formatted_color_maps = holes.matplotlib_formatted_color_maps

        self.categorical_vars = holes.categorical_vars
        self.continuous_vars = holes.continuous_vars

        intervals_mesh = holes.make_intervals_mesh("intervals")
        self.add_intervals(intervals_mesh, *args, **kwargs)

    @property
    def hole_vars(self):
        """Return the variables corresponding to the scalar data attached to the hole interval cells.

        Returns
        -------
        list[str]
            List of variable names
        """
        return self._hole_vars

    @property
    def cmaps(self):
        return self._cmaps

    @property
    def faces_per_interval(self):
        return self._faces_per_interval

    def _make_selectable(
        self,
        actor,
        selection_color="magenta",
        accelerated_selection=False,
    ):
        self.selection_color = selection_color
        # make pickable
        # actor.SetPickable = 0

        # track clicks
        self.track_click_position(side="left", callback=self._make_selection)

        # set selection actor
        self.selection_actor = None

        self._picked_interval_cell = None
        self._selected_intervals = []
        self._selected_interval_cells = []

        # set actor picker
        self.actor_picker = vtkPropPicker()

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

        self.drillhole_interval_cell_picker = cell_picker

    def _make_selection(self, *args):
        pos = self.click_position + (0,)
        actor_picker = self.actor_picker
        actor_picker.Pick(pos[0], pos[1], pos[2], self.renderer)
        picked_actor = actor_picker.GetActor()
        if picked_actor is not None:
            if picked_actor == self._actors["drillhole intervals"]:
                cell_picker = self.drillhole_interval_cell_picker
                cell_picker.Pick(pos[0], pos[1], pos[2], self.renderer)
                picked_interval_cell = cell_picker.GetCellId()

                if picked_interval_cell is not None:
                    shift_pressed = self.iren.interactor.GetShiftKey()
                    ctrl_pressed = self.iren.interactor.GetControlKey()

                    if shift_pressed == True:
                        self._make_continuous_multi_selection(picked_interval_cell)
                    elif ctrl_pressed == True:
                        self._make_discontinuous_multi_selection(picked_interval_cell)
                    else:
                        self._make_single_selection(picked_interval_cell)

                    self._picked_interval_cell = picked_interval_cell

                    self._update_selection_object(
                        "interval", self._selected_interval_cells
                    )

    def _make_single_selection(self, picked_interval_cell):
        selected_interval = int(
            np.floor(picked_interval_cell / self.faces_per_interval)
        )
        selected_interval_cells = np.arange(
            selected_interval * self.faces_per_interval,
            (selected_interval + 1) * self.faces_per_interval,
        ).tolist()

        self._selected_intervals = [selected_interval]
        self._selected_interval_cells = selected_interval_cells

        return self._selected_intervals, self._selected_interval_cells

    def _make_discontinuous_multi_selection(self, picked_interval_cell):
        selected_interval = int(
            np.floor(picked_interval_cell / self.faces_per_interval)
        )
        selected_interval_cells = np.arange(
            selected_interval * self.faces_per_interval,
            (selected_interval + 1) * self.faces_per_interval,
        ).tolist()

        self._selected_intervals += [selected_interval]
        self._selected_interval_cells += selected_interval_cells

        return self._selected_intervals, self._selected_interval_cells

    def _make_continuous_multi_selection(self, picked_interval_cell):
        if (
            self._picked_interval_cell < picked_interval_cell
        ):  # normal direction (down the hole)
            selected_intervals = np.arange(
                self._selected_intervals[-1] + 1,
                int(np.floor(picked_interval_cell / self.faces_per_interval)) + 1,
            ).tolist()
            selected_interval_cells = np.arange(
                (selected_intervals[0]) * self.faces_per_interval,
                (selected_intervals[-1] + 1) * self.faces_per_interval,
            ).tolist()

            self._selected_intervals += selected_intervals
            self._selected_interval_cells += selected_interval_cells
        else:  # reverse direction (up the hole)
            selected_intervals = np.arange(
                int(np.floor(picked_interval_cell / self.faces_per_interval)),
                self._selected_intervals[-1],
            ).tolist()
            selected_interval_cells = np.arange(
                (selected_intervals[0] * self.faces_per_interval),
                (selected_intervals[-1] + 1) * self.faces_per_interval,
            ).tolist()

            self._selected_intervals = selected_intervals + self._selected_intervals
            self._selected_interval_cells = (
                selected_interval_cells + self._selected_interval_cells
            )

        return self._selected_intervals, self._selected_interval_cells

    def _update_selection_object(self, interval_or_sample, selected_cells):
        mesh = self._filters["drillhole intervals"]
        sel_mesh = mesh.extract_cells(selected_cells)
        self.selection_actor = self.add_mesh(
            sel_mesh,
            name="drillhole intervals selection",
            color=self.selection_color,
            reset_camera=False,
            pickable=False,
        )
        self.selection_actor.mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
            0, -5
        )
        self.render()

    # def _filter_intervals(self, filtered_cells):
    #     mesh = self._filters["drillhole intervals"]
    #     self._unfiltered_mesh = mesh
    #     self.remove_actor(self._actors["drillhole intervals"])

    #     filt_mesh = mesh.extract_cells(filtered_cells)
    #     filt_mesh = filt_mesh.extract_geometry(filt_mesh.bounds)
    #     self.filtered_mesh = filt_mesh
    #     self.add_holes(filt_mesh)

    @property
    def active_var(self):
        return self._active_var

    @active_var.setter
    def active_var(self, active_var):
        if hasattr(self, "_prev_active_var"):
            self._prev_active_var = self.active_var
        else:
            self._prev_active_var = None

        self._active_var = active_var
        self._actors["drillhole intervals"].mapper.dataset.set_active_scalars(
            active_var
        )
        if active_var in self.categorical_vars:
            self.cmap = self.matplotlib_formatted_color_maps[active_var]
            self.cmap_range = (0, list(self.code_to_cat_map[active_var].keys())[-1])
        else:
            if self._prev_active_var in self.categorical_vars:
                self.cmap = self.continuous_cmap
                self.reset_cmap_range()
            else:
                self.reset_cmap_range()
        self.render()

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        self._cmap = cmap
        if self.active_var in self.continuous_vars:
            self._actors["drillhole intervals"].mapper.lookup_table.cmap = cmap
            self._actors[
                "drillhole intervals"
            ].mapper.lookup_table.nan_color = "white"  # self.nan_opacity
            self.continuous_cmap = cmap

        else:
            self._actors["drillhole intervals"].mapper.lookup_table = pv.LookupTable(
                cmap
            )
            self._actors[
                "drillhole intervals"
            ].mapper.lookup_table.nan_color = "white"  # self.nan_opacity

        self.render()

    @property
    def cmap_range(self):
        return self._cmap_range

    @cmap_range.setter
    def cmap_range(self, cmap_range):
        self._cmap_range = cmap_range
        self._actors[
            "drillhole intervals"
        ].mapper.lookup_table.scalar_range = cmap_range
        self._actors["drillhole intervals"].mapper.SetUseLookupTableScalarRange(True)
        self.render()

    def reset_cmap_range(self):
        mesh = self._filters["drillhole intervals"]
        array = mesh.cell_data[self.active_var]
        min, max = array.min(), array.max()
        self.cmap_range = (min, max)

        return self.cmap_range

    def update_visibility(self, visible, actor_name):
        if visible == True:
            self._actors[actor_name].prop.opacity = 1
        else:
            self._actors[actor_name].prop.opacity = 0

        self.render()

    def update_opacity(self, opacity, actor_name):
        self._actors[actor_name].prop.opacity = opacity

        self.render()

    def get_assay_data(self):
        intervals = self._selected_intervals

        holes_mesh = self._filters["drillhole intervals"]
        vars = holes_mesh.array_names
        assay_dict = {}
        for var in vars:
            assay_dict[var] = holes_mesh[var][:: self.faces_per_interval][intervals]

        assay_dict.pop("TubeNormals")  # pandas won't except columns that aren't 1D
        assay = pd.DataFrame(assay_dict)

        return assay

    @property
    def selected_intervals(self):
        intervals = self._selected_intervals

        holes_mesh = self._filters["drillhole intervals"]
        vars = holes_mesh.array_names
        assay_dict = {}
        for var in vars:
            assay_dict[var] = holes_mesh[var][:: self.faces_per_interval][intervals]

        assay_dict.pop("TubeNormals")  # pandas won't except columns that aren't 1D
        assay = pd.DataFrame(assay_dict)

        return assay

    @selected_intervals.setter
    def selected_intervals(self, intervals):
        interval_cells = []
        for interval in intervals:
            interval_cells += np.arange(
                interval * self.faces_per_interval,
                (interval + 1) * self.faces_per_interval,
            ).tolist()

        self._selected_intervals = intervals
        self._selected_interval_cells = interval_cells

        self._update_selection_object("interval", self._selected_interval_cells)

    # @property
    # def filtered_intervals(self):
    #     return self._filtered_intervals

    # @filtered_intervals.setter
    # def filtered_intervals(self, intervals):
    #     interval_cells = []
    #     for interval in intervals:
    #         interval_cells += np.arange(interval * self.faces_per_interval, (interval + 1) * self.faces_per_interval).tolist()
    #     self._filtered_intervals = intervals
    #     self._filtered_interval_cells = interval_cells

    #     self._filter_intervals(self._filtered_interval_cells)

    # def reset_filter(self):
    #     self.remove_actor(self._actor["drillhole intervals"]);;.;l,;.
    #     self.add_holes(self._unfiltered_mesh)

    def selected_drill_log(
        self, categorical_interval_vars=None, continuous_interval_vars=None
    ):
        data = self.selected_intervals
        if categorical_interval_vars is None:
            categorical_interval_vars = self.categorical_vars

        if continuous_interval_vars is None:
            continuous_interval_vars = self.continuous_vars

        log = DrillLog()
        depths = data[["from", "to"]].values

        for var in categorical_interval_vars:
            values = data[var].values
            log.add_categorical_interval_data(
                var,
                depths,
                values,
                self.code_to_cat_map[var],
                self.code_to_color_map[var],
            )
        for var in continuous_interval_vars:
            values = data[var].values
            log.add_continuous_interval_data(var, depths, values)

        log.create_figure()

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

    def add_mesh(self, mesh, name, add_show_widgets=True, *args, **kwargs):
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
        actor = super(DrillDownPanelPlotter, self).add_mesh(mesh, name, *args, **kwargs)

        if name == "drillhole intervals selection":
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
            # self.mesh_opacity_widgets.append(opacity_widget)
            opacity_widget.param.watch(
                partial(self._on_mesh_opacity_change, name), "value"
            )

            # situate show and opacity widgets; add to mesh visibility widget
            self.mesh_visibility_card.append(pn.pane.Markdown(f"{name}"))
            self.mesh_visibility_card.append(pn.Row(show_widget, opacity_widget))
            self.mesh_visibility_card.append(pn.layout.Divider())

        return actor

    def add_intervals(
        self,
        mesh,
        active_var=None,
        cmap="blues",
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
            Matplotlib color map used to color interval data. By default "blues"
        cmap_range : tuple, optional
            Minimum and maximum value between which color map is applied. By default None
        """
        tube_mesh = super(DrillDownPanelPlotter, self).add_intervals(
            mesh,
            active_var=active_var,
            cmap=cmap,
            cmap_range=cmap_range,
            *args,
            **kwargs,
        )

        self._make_hole_ctrl_card(active_var, cmap, cmap_range)
        # set up widget to show and hide mesh
        if active_var is not None:
            self.active_var = active_var
        self.cmap = cmap
        if cmap_range != None:
            self.cmap_range = cmap_range

    def _make_active_var_widget(self, value=None):
        widget = pn.widgets.Select(
            name="active variable",
            options=self.hole_vars,
            value=value,
            width=int(0.9 * self.ctrl_widget_width),
        )
        widget.param.watch(self._on_active_var_change, "value")

        self.active_var_widget = widget
        return widget

    def _make_cmap_widget(self, value=None):
        widget = pn.widgets.Select(
            name="color map",
            options=self.cmaps,
            value=value,
            width=int(0.9 * self.ctrl_widget_width),
        )
        widget.param.watch(self._on_cmap_change, "value")

        self.cmap_widget = widget
        return widget

    def _make_cmap_range_widget(self, value=None):
        min = self._actors["drillhole intervals"].mapper.dataset.active_scalars.min()
        max = self._actors["drillhole intervals"].mapper.dataset.active_scalars.max()
        if value == None:
            value = (min, max)
        widget = pn.widgets.RangeSlider(
            name="color map range",
            start=min,
            end=max,
            step=min - max / 1000,
            value=value,
            width=int(0.9 * self.ctrl_widget_width),
        )
        widget.param.watch(self._on_cmap_range_change, "value")
        widget.param.default = value

        self.cmap_range_widget = widget
        return widget

    def _make_hole_ctrl_card(self, active_var=None, cmap=None, cmap_range=None):
        self.hole_ctrl_card = pn.Column(
            width=self.ctrl_widget_width,
        )
        self.hole_ctrl_card.append(self._make_active_var_widget(value=active_var))
        self.hole_ctrl_card.append(self._make_cmap_widget(value=cmap))
        self.hole_ctrl_card.append(self._make_cmap_range_widget(value=cmap_range))

        self.ctrls.append(("hole controls", self.hole_ctrl_card))
        return self.hole_ctrl_card

    def _make_mesh_visibility_card(self):
        self.mesh_visibility_card = pn.Column(scroll=True, width=self.ctrl_widget_width)

        return self.mesh_visibility_card

    @property
    def active_var(self):
        return self._active_var

    @active_var.setter
    def active_var(self, active_var):
        super(DrillDownPanelPlotter, DrillDownPanelPlotter).active_var.fset(
            self, active_var
        )
        if hasattr(self, "active_var_widget"):
            self.active_var_widget.value = active_var

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        super(DrillDownPanelPlotter, DrillDownPanelPlotter).cmap.fset(self, cmap)
        if isinstance(cmap, str):
            if hasattr(self, "cmap_widget"):
                self.cmap_widget.value = cmap
                self.cmap_widget.visible = True
                self.cmap_range_widget.visible = True
        else:
            self.cmap_widget.visible = False
            self.cmap_range_widget.visible = False

    @property
    def cmap_range(self):
        return self._cmap_range

    @cmap_range.setter
    def cmap_range(self, cmap_range):
        super(DrillDownPanelPlotter, DrillDownPanelPlotter).cmap_range.fset(
            self, cmap_range
        )
        if hasattr(self, "cmap_range_widget"):
            self.cmap_range_widget.value = cmap_range
            self.cmap_range_widget.step = (cmap_range[1] - cmap_range[0]) / 1000

    def update_visibility(self, visible, actor_name):
        super(DrillDownPanelPlotter, self).update_visibility(visible, actor_name)
        if hasattr(self, "show_widgets"):
            self.show_widgets[actor_name].value = visible

    def update_opacity(self, opacity, actor_name):
        super(DrillDownPanelPlotter, self).update_opacity(opacity, actor_name)
        if hasattr(self, "opacity_widgets"):
            self.opacity_widgets[actor_name].value = opacity

    def _on_active_var_change(self, event):
        active_var = event.new
        self.active_var = active_var

        active_scalars = self._actors[
            "drillhole intervals"
        ].mapper.dataset.active_scalars
        self.cmap_range_widget.start = active_scalars.min()
        self.cmap_range_widget.end = active_scalars.max()

    def _on_cmap_change(self, event):
        cmap = event.new
        self.cmap = cmap

        active_scalars = self._actors[
            "drillhole intervals"
        ].mapper.dataset.active_scalars
        self.cmap_range_widget.start = active_scalars.min()
        self.cmap_range_widget.end = active_scalars.max()

    def _on_cmap_range_change(self, event):
        cmap_range = event.new
        self.cmap_range = cmap_range

    def _on_mesh_show_change(self, actor_name, event):
        visible = event.new
        self.update_visibility(visible, actor_name=actor_name)

    def _on_mesh_opacity_change(self, actor_name, event):
        opacity = event.new
        self.update_opacity(opacity, actor_name=actor_name)

    def get_assay_data(self):
        return super(DrillDownPanelPlotter, self).get_assay_data()

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

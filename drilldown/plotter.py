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
)
import vtk.util.numpy_support as vtknumpy
import pyvista as pv
from pyvista import Plotter
import numpy as np
import pandas as pd
from IPython.display import IFrame
import panel as pn
from matplotlib import pyplot as plt
from functools import partial

from pyvista.trame.jupyter import show_trame
from .drill_log import DrillLog


class DrillDownPlotter(Plotter):
    """Plotting object for displaying drillholes and related datasets."""

    def __init__(self, *args, **kwargs):
        """Initialize plotter."""

        super().__init__(*args, **kwargs)

        self.set_background("white")
        self.enable_trackball_style()

        self._actors = {}
        self._filters = {}
        self.selection_extracts = {}

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

        actor = super(DrillDownPlotter, self).add_mesh(mesh, *args, **kwargs)
        actor.SetPickable = 0
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
            *args,
            **kwargs,
        )

    def add_surveys(self, mesh, *args, **kwargs):
        name = "surveys"
        self.add_mesh(mesh, name, *args, **kwargs)

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
        color_on_selection="#000000",
        opacity_on_selection=1,
        accelerated_selection=False,
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
        color_on_selection : ColorLike, optional
            Color used to color the selection object. By default "#000000"
        opacity_on_selection : float, optional
            Opacity used for the selection object. By default 1
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

        self.update_cmap(cmap)
        if cmap_range != None:
            self.update_cmap_range(cmap_range)

        if selectable == True:
            self._make_selectable(
                actor,
                color_on_selection=color_on_selection,
                opacity_on_selection=opacity_on_selection,
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
        color_on_selection="#000000",
        opacity_on_selection=1,
        accelerated_selection=False,
    ):
        # make pickable
        actor.SetPickable = 0

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
                        self._make_multi_selection(
                            picked_interval_cell, continuous=True
                        )
                    elif ctrl_pressed == True:
                        self._make_multi_selection(
                            picked_interval_cell, continuous=False
                        )
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

    def _make_multi_selection(self, picked_interval_cell, continuous=False):
        if continuous == True:
            (
                selected_intervals,
                selected_interval_cells,
            ) = self._make_continuous_multi_selection(picked_interval_cell)
        else:
            (
                selected_intervals,
                selected_interval_cells,
            ) = self._make_discontinuous_multi_selection(picked_interval_cell)

        self._selected_intervals += selected_intervals
        self._selected_interval_cells += selected_interval_cells

        return self._selected_intervals, self._selected_interval_cells

    def _make_discontinuous_multi_selection(self, picked_interval_cell):
        selected_interval = int(
            np.floor(picked_interval_cell / self.faces_per_interval)
        )
        selected_interval_cells = np.arange(
            selected_interval * self.faces_per_interval,
            (selected_interval + 1) * self.faces_per_interval,
        ).tolist()

        return [selected_interval], selected_interval_cells

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

        else:  # reverse direction (up the hole)
            selected_intervals = np.arange(
                int(np.floor(picked_interval_cell / self.faces_per_interval)),
                self._selected_intervals[-1],
            ).tolist()
            selected_interval_cells = np.arange(
                (selected_intervals[0] * self.faces_per_interval),
                (selected_intervals[-1] + 1) * self.faces_per_interval,
            ).tolist()

        return selected_intervals, selected_interval_cells

    def _update_selection_object(self, interval_or_sample, selected_cells):
        mesh = self._filters["drillhole intervals"]
        sel_mesh = mesh.extract_cells(selected_cells)

        sel_mapper = pv.DataSetMapper(sel_mesh)
        sel_mapper.SetResolveCoincidentTopologyToPolygonOffset()
        sel_mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(0.0, -0.5)

        sel_actor = pv.Actor(mapper=sel_mapper)
        sel_actor.prop.color = "magenta"

        if self.selection_actor:
            self.remove_actor(self.selection_actor)

        self.selection_actor, _ = self.add_actor(
            sel_actor,
            name="drillhole intervals selection",
            reset_camera=False,
            pickable=False,
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

    def update_active_var(self, active_var):
        self._actors["drillhole intervals"].mapper.dataset.set_active_scalars(
            active_var
        )
        self.render()

    def update_cmap(self, cmap):
        self._actors["drillhole intervals"].mapper.lookup_table.cmap = cmap
        self.render()

    def update_cmap_range(self, cmap_range):
        self._actors[
            "drillhole intervals"
        ].mapper.lookup_table.scalar_range = cmap_range
        self._actors["drillhole intervals"].mapper.SetUseLookupTableScalarRange(True)
        self.render()

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

    def selected_drill_log(self):
        data = self.selected_intervals
        log = DrillLog()
        depths = data[["from", "to"]].values
        vars = [
            var
            for var in data.columns
            if var
            not in [
                "from",
                "to",
                "vtkOriginalPointIds",
                "vtkOriginalCellIds",
                "hole ID",
            ]
        ]
        for var in vars:
            values = data[var].values
            log.add_continuous_interval_data(depths, values, var)

        log.create_figure()

        return log.fig

    def iframe(self, w=800, h=400):
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
            width=300,
        )

        super(DrillDownPanelPlotter, self).__init__(*args, **kwargs)

        super(pn.Row, self).__init__(
            self.ctrls, self.iframe(sizing_mode="stretch_both"), height=800, width=1200
        )

    def add_mesh(self, mesh, name, *args, **kwargs):
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
        opacity_widget.param.watch(partial(self._on_mesh_opacity_change, name), "value")

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

        super(DrillDownPanelPlotter, self).add_intervals(
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
            self.update_active_var(active_var)
        self.update_cmap(cmap)
        if cmap_range != None:
            self.update_cmap_range(cmap_range)

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

    def update_active_var(self, active_var):
        super(DrillDownPanelPlotter, self).update_active_var(active_var)
        if hasattr(self, "active_var_widget"):
            self.active_var_widget.value = active_var

    def update_cmap(self, cmap):
        super(DrillDownPanelPlotter, self).update_cmap(cmap)
        if hasattr(self, "cmap_widget"):
            self.cmap_widget.value = cmap

    def update_cmap_range(self, cmap_range):
        super(DrillDownPanelPlotter, self).update_cmap_range(cmap_range)
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
        self.update_active_var(active_var)

        active_scalars = self._actors[
            "drillhole intervals"
        ].mapper.dataset.active_scalars
        self.cmap_range_widget.start = active_scalars.min()
        self.cmap_range_widget.end = active_scalars.max()

    def _on_cmap_change(self, event):
        cmap = event.new
        self.update_cmap(cmap)

        active_scalars = self._actors[
            "drillhole intervals"
        ].mapper.dataset.active_scalars
        self.cmap_range_widget.start = active_scalars.min()
        self.cmap_range_widget.end = active_scalars.max()

    def _on_cmap_range_change(self, event):
        cmap_range = event.new
        self.update_cmap_range(cmap_range)

    def _on_mesh_show_change(self, actor_name, event):
        visible = event.new
        self.update_visibility(visible, actor_name=actor_name)

    def _on_mesh_opacity_change(self, actor_name, event):
        opacity = event.new
        self.update_opacity(opacity, actor_name=actor_name)

    def get_assay_data(self):
        return super(DrillDownPanelPlotter, self).get_assay_data()

    def iframe(self, sizing_mode="fixed", w=800, h=400):
        _iframe = super(DrillDownPanelPlotter, self).iframe()
        _src = _iframe.src

        if sizing_mode == "stretch_width":
            w = "100%"
        elif sizing_mode == "stretch_height":
            h = "100%"
        elif sizing_mode == "stretch_both":
            w = "100%"
            h = "100%"

        html = f"""<iframe frameborder="0" title="panel app" style="width: {w};height: {h};flex-grow: 1" src="{_src}"></iframe>"""
        _iframe = pn.pane.HTML(html, sizing_mode=sizing_mode)

        return pn.Column(
            pn.panel(_iframe, sizing_mode=sizing_mode), sizing_mode=sizing_mode
        )

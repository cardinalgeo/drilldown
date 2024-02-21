from trame.widgets import vuetify, markdown

from trame.ui.vuetify import SinglePageWithDrawerLayout
from trame.app import get_server
from trame_server.utils.browser import open_browser

from pyvista.trame import PyVistaRemoteView
from pyvista.trame.jupyter import elegantly_launch

import panel as pn
from functools import partial
import numpy as np

from ..plotter import DrillDownPlotter
from ..utils import is_jupyter
from ..layer.layer import IntervalDataLayer, PointDataLayer
from .layer_list import LayerListUI


def ui_card(title, ui_name):
    with vuetify.VCard():
        vuetify.VCardSubtitle(
            title,
            classes="grey lighten-1 py-1 grey--text text--darken-3",
            style="user-select: none; cursor: pointer",
            hide_details=True,
            dense=True,
        )
        content = vuetify.VCardText(classes="py-2")
    return content


class DrillDownTramePlotter(DrillDownPlotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ui = None
        self.layer_list_ui = None
        self.state.ctrl_mesh_name = 0

    def show(self, inline=False, return_viewer=False):
        if return_viewer == True:
            viewer = super().show(return_viewer=True)

            return viewer

        self._ui = self._initialize_ui()
        self._initialize_engine()

        if inline == True:
            if is_jupyter():
                elegantly_launch(self.server)  # launch server in nb w/o using await

                return self._ui

            else:
                raise ValueError("Inline mode only available in Jupyter notebook.")

        else:
            if is_jupyter():
                self.server.start(
                    exec_mode="task",
                    port=0,
                    open_browser=True,
                )
            else:
                self.server.start(
                    port=0,
                    open_browser=True,
                )

            ctrl = self.server.controller
            ctrl.on_server_ready.add(lambda **kwargs: open_browser(self.server))

    def _initialize_ui(self):
        with SinglePageWithDrawerLayout(self.server) as layout:
            layout.title.set_text("")
            with layout.footer as footer:
                footer.clear()
                footer.style = "z-index: 1000"
                md = markdown.Markdown()
                md.update("DrillDown")
            with layout.content:
                with vuetify.VContainer(
                    fluid=True,
                    classes="fill-height pa-0 ma-0",
                ):
                    PyVistaRemoteView(self)

            with layout.drawer as drawer:
                with vuetify.VContainer(
                    classes="fill-height pa-0 ma-0", style="overflow: hidden;"
                ) as self.drawer_content:
                    with vuetify.VCard(
                        elevation=0,
                        style="background-color: #f5f5f5; overflow: hidden; height: 40%; width: 90%; margin-left: auto; margin-right: auto; margin-top: auto; margin-bottom: auto;",
                    ):

                        vuetify.VSlider(
                            hide_details=True,
                            label="opacity",
                            v_model=("opacity",),
                            max=1,
                            min=0,
                            step=0.001,
                            style="width: 90%; margin-left: auto; margin-right: auto",
                        )
                        if (isinstance(self.layers[-1], IntervalDataLayer)) or (
                            isinstance(self.layers[-1], PointDataLayer)
                        ):
                            vuetify.VDivider(
                                classes="mb-2",
                                v_show=("divider_visible", True),
                                style="width: 90%; margin-left: auto; margin-right: auto",
                            )
                            visible = True
                        else:
                            visible = False

                        vuetify.VSelect(
                            label="active array name",
                            v_show=("active_array_name_visible", visible),
                            v_model=(
                                "active_array_name",
                                self.layers[-1].active_array_name,
                            ),
                            items=("array_names",),
                            classes="pt-1",
                            style="width: 90%; margin-left: auto; margin-right: auto",
                            # **DROPDOWN_STYLES,
                        )

                        if (visible == False) or (
                            self.state.active_array_name
                            in self.layers[-1].categorical_array_names
                        ):
                            visible = False

                        vuetify.VSelect(
                            label="colormap",
                            v_show=("cmap_visible", visible),
                            v_model=("cmap",),
                            items=("cmap_fields",),
                            classes="pt-1",
                            style="width: 90%; margin-left: auto; margin-right: auto",
                        )

                        vuetify.VRangeSlider(
                            label="colormap limits",
                            v_show=("clim_visible", visible),
                            v_model=("clim",),
                            min=("clim_min",),
                            max=("clim_max",),
                            step=("clim_step",),
                            classes="pt-1",
                            style="width: 90%; margin-left: auto; margin-right: auto",
                        )

                    self.layer_list_ui = LayerListUI()

        for layer in self.layers:
            self.layer_list_ui.add_layer(layer)

        layout.flush_content()
        self.state.flush()

        return layout

    def _initialize_engine(self):
        state = self.server.state

        @state.change("ctrl_mesh_name")
        def update_controls_mesh(ctrl_mesh_name, **kwargs):
            layer = self.layers[ctrl_mesh_name]

            # update active var
            state.active_array_name_visible = True
            state.array_names = layer.array_names
            state.active_array_name = layer.active_array_name

            # update cmap
            if state.active_array_name in layer.continuous_array_names:
                state.cmap_visible = True
                state.cmap_fields = layer._cmaps
                state.cmap = layer.cmap

                state.clim_visible = True
                state.clim_step = layer.clim_step

                state.clim = layer.clim
                state.clim_min = layer.clim_range[0]
                state.clim_max = layer.clim_range[1]

            else:
                state.cmap_visible = False
                state.clim_visible = False

            state.visibility = layer.visibility
            state.opacity = layer.opacity

        @state.change("visibility")
        def update_visibility(visibility, **kwargs):
            layer = self.layers[state.ctrl_mesh_name]

            if visibility != layer.visibility:
                layer.visibility = visibility

        @state.change("opacity")
        def update_opacity(opacity, **kwargs):
            layer = self.layers[state.ctrl_mesh_name]

            if opacity != layer.opacity:
                layer.opacity = opacity

        @state.change("active_array_name")
        def update_active_array_name(active_array_name, **kwargs):
            layer = self.layers[state.ctrl_mesh_name]

            if active_array_name != layer.active_array_name:
                layer.active_array_name = active_array_name

            if active_array_name in layer.continuous_array_names:
                state.cmap_visible = True
                state.clim_visible = True

            else:
                state.cmap_visible = False
                state.clim_visible = False

        @state.change("cmap")
        def update_cmap(cmap, **kwargs):
            layer = self.layers[state.ctrl_mesh_name]

            if cmap != layer.cmap:
                layer.cmap = cmap

        @state.change("clim")
        def update_clim(clim, **kwargs):
            layer = self.layers[state.ctrl_mesh_name]
            if (abs(clim[0] - layer.clim[0]) > 1e-6) or (
                abs(clim[1] - layer.clim[1]) > 1e-6
            ):
                layer.clim = clim

    def add_intervals(self, *args, **kwargs):
        super().add_intervals(*args, **kwargs)
        if self.layer_list_ui is not None:
            self.layer_list_ui.add_layer(self.layers[-1])
            self._ui.flush_content()
            self.state.flush()

    def add_points(self, *args, **kwargs):
        super().add_points(*args, **kwargs)
        if self.layer_list_ui is not None:
            self.layer_list_ui.add_layer(self.layers[-1])
            self._ui.flush_content()
            self.state.flush()


class DrillDownPanelPlotter(DrillDownPlotter, pn.Row):
    """Plotting object for displaying drillholes and related datasets with simple GUI."""

    def __init__(self, *args, **kwargs):
        """Initialize plotter."""
        self.ctrl_widget_width = 300
        self.active_var_widgets = {}
        self.cmap_widgets = {}
        self.clim_widgets = {}
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
        clim=None,
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
            clim=clim,
            selection_color=selection_color,
            accelerated_selection=accelerated_selection,
            nan_opacity=nan_opacity,
            *args,
            **kwargs,
        )

        self._make_hole_ctrl_card(name, active_var, cmap, clim)

        return actor

    def add_intervals_mesh(
        self,
        mesh,
        name=None,
        categorical_vars=[],
        continuous_vars=[],
        active_var=None,
        cmap="Blues",
        clim=None,
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
            Matplotlib colormap used to color interval data. By default "Blues"
        clim : tuple, optional
            Minimum and maximum value between which colormap is applied. By default None
        """

        super(DrillDownPanelPlotter, self).add_intervals_mesh(
            mesh,
            name=name,
            active_var=active_var,
            categorical_vars=categorical_vars,
            continuous_vars=continuous_vars,
            cmap=cmap,
            clim=clim,
            *args,
            **kwargs,
        )
        # set up widget to show and hide mesh
        if active_var is not None:
            self.active_var = (name, active_var)
        self.cmap = (name, cmap)
        if clim != None:
            self.clim = (name, clim)

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
        clim=None,
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
            clim=clim,
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
        if clim != None:
            self._clim[name] = clim

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
            name=f"{name} colormap",
            options=self.cmaps,
            value=cmap,
            width=int(0.9 * self.ctrl_widget_width),
        )
        widget.param.watch(partial(self._on_cmap_change, name), "value")

        self.cmap_widgets[name] = widget
        return widget

    def _make_clim_widget(self, name, clim=None):
        min = self.actors[name].mapper.dataset.active_scalars.min()
        max = self.actors[name].mapper.dataset.active_scalars.max()
        if clim == None:
            clim = (min, max)
        widget = pn.widgets.RangeSlider(
            name=f"{name} colormap range",
            start=min,
            end=max,
            step=min - max / 1000,
            value=clim,
            width=int(0.9 * self.ctrl_widget_width),
        )
        widget.param.watch(partial(self._on_clim_change, name), "value")
        widget.param.default = clim

        self.clim_widgets[name] = widget
        return widget

    def _make_hole_ctrl_card(self, name, active_var=None, cmap="Blues", clim=None):
        self.hole_ctrl_card = pn.Column(
            width=self.ctrl_widget_width,
        )
        self.hole_ctrl_card.append(self._make_active_var_widget(name, active_var))
        self.hole_ctrl_card.append(self._make_cmap_widget(name, cmap))
        self.hole_ctrl_card.append(self._make_clim_widget(name, clim))

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
                self.clim_widgets[name].visible = True

            elif active_var in self.categorical_vars[name]:
                self.cmap_widgets[name].visible = False
                self.clim_widgets[name].visible = False

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, key_value_pair):
        super(DrillDownPanelPlotter, DrillDownPanelPlotter).cmap.fset(
            self, key_value_pair
        )
        name, cmap = key_value_pair
        if (name in self.cmap_widgets.keys()) and (name in self.clim_widgets.keys()):
            if isinstance(cmap, str):
                self.cmap_widgets[name].value = cmap

    @property
    def clim(self):
        return self._clim

    @clim.setter
    def clim(self, key_value_pair):
        super(DrillDownPanelPlotter, DrillDownPanelPlotter).clim.fset(
            self, key_value_pair
        )
        name, clim = key_value_pair
        if name in self.clim_widgets.keys():
            self.clim_widgets[name].value = clim
            self.clim_widgets[name].step = (clim[1] - clim[0]) / 1000

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
        self.clim_widgets[name].start = active_scalars.min()
        self.clim_widgets[name].end = active_scalars.max()

    def _on_cmap_change(self, name, event):
        cmap = event.new
        self.cmap = (name, cmap)

        active_scalars = self.actors[name].mapper.dataset.active_scalars
        self.clim_widgets[name].start = active_scalars.min()
        self.clim_widgets[name].end = active_scalars.max()

    def _on_clim_change(self, name, event):
        clim = event.new
        self.clim = (name, clim)

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

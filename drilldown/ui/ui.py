from trame.widgets import vuetify, markdown

from trame.ui.vuetify import SinglePageWithDrawerLayout
from trame.app import get_server
from trame_server.utils.browser import open_browser

from pyvista.trame import PyVistaRemoteView
from pyvista.trame.jupyter import elegantly_launch

import panel as pn
from functools import partial

from ..plotter import DrillDownPlotter


def is_jupyter():
    from IPython import get_ipython

    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


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
        self.server = get_server()
        self.state = self.server.state
        self.server.client_type = "vue2"

        self._ui = None

    def show(self, inline=True):
        self._ui = self._initialize_ui()
        self._initialize_engine()

        if inline == True:
            elegantly_launch(self.server)  # launch server in nb w/o using await

            return self._ui

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
        ctrl = self.server.controller
        ctrl_mesh_name = self.mesh_names[-1]
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
                with ui_card("Controls", "controls"):
                    vuetify.VSelect(
                        label="controls mesh",
                        v_model=("ctrl_mesh_name", ctrl_mesh_name),
                        items=("ctrl_mesh_name_fields", self.mesh_names.copy()),
                        classes="pt-1",
                        # **DROPDOWN_STYLES,
                    )
                    vuetify.VDivider(classes="mb-2")

                    vuetify.VSlider(
                        hide_details=True,
                        label="opacity",
                        v_model=("opacity", 1),
                        max=1,
                        min=0,
                        step=0.001,
                        style="max-width: 300px;",
                    )
                    if (
                        ctrl_mesh_name
                        in self.point_actor_names + self.interval_actor_names
                    ):
                        vuetify.VDivider(
                            classes="mb-2", v_show=("divider_visible", True)
                        )

                        vuetify.VSelect(
                            label="active variable",
                            v_show=("active_var_visible", True),
                            v_model=("active_var", self.active_var[ctrl_mesh_name]),
                            items=(
                                "active_var_fields",
                                self.all_vars[ctrl_mesh_name],
                            ),
                            classes="pt-1",
                            # **DROPDOWN_STYLES,
                        )
                        vuetify.VSelect(
                            label="color map",
                            v_show=("cmap_visible", True),
                            v_model=("cmap", self.cmap[ctrl_mesh_name]),
                            items=("cmap_fields", self.cmaps),
                            classes="pt-1",
                            # **DROPDOWN_STYLES,
                        )
                        vuetify.VRangeSlider(
                            label="colormap range",
                            v_show=("cmap_range_visible", True),
                            v_model=("cmap_range", self.cmap_range[ctrl_mesh_name]),
                            min=("cmap_range_min", self.cmap_range[ctrl_mesh_name][0]),
                            max=("cmap_range_max", self.cmap_range[ctrl_mesh_name][1]),
                            classes="pt-1",
                            # **DROPDOWN_STYLES
                        )

                    else:
                        vuetify.VDivider(
                            classes="mb-2", v_show=("divider_visible", False)
                        )

                        vuetify.VSelect(
                            label="active variable",
                            v_show=("active_var_visible", False),
                            v_model=("active_var", None),
                            items=(
                                "active_var_fields",
                                None,
                            ),
                            classes="pt-1",
                            # **DROPDOWN_STYLES,
                        )
                        vuetify.VSelect(
                            label="color map",
                            v_show=("cmap_visible", False),
                            v_model=("cmap", None),
                            items=("cmap_fields", None),
                            classes="pt-1",
                            # **DROPDOWN_STYLES,
                        )
                        vuetify.VRangeSlider(
                            label="colormap range",
                            v_show=("cmap_range_visible", False),
                            v_model=("cmap_range", (0, 1)),
                            min=("cmap_range_min", 0),
                            max=("cmap_range_max", 1),
                            classes="pt-1",
                            # **DROPDOWN_STYLES
                        )
        return layout

    def _initialize_engine(self):
        state = self.server.state

        @state.change("ctrl_mesh_name")
        def update_controls_mesh(ctrl_mesh_name, **kwargs):
            if ctrl_mesh_name not in self.interval_actor_names + self.point_actor_names:
                state.divider_visible = False
                state.active_var_visible = False
                state.cmap_visible = False
                state.cmap_range_visible = False

            else:
                state.divider_visible = True

                # update active var
                state.active_var_visible = True
                state.active_var = self.active_var[ctrl_mesh_name]
                if ctrl_mesh_name in self.interval_actor_names:
                    state.active_var_fields = self.interval_vars[ctrl_mesh_name]
                elif ctrl_mesh_name in self.point_actor_names:
                    state.active_var_fields = self.point_vars[ctrl_mesh_name]

                # update cmap
                if state.active_var in self.continuous_vars[ctrl_mesh_name]:
                    state.cmap_visible = True
                    state.cmap = self.cmap[ctrl_mesh_name]
                    state.cmap_fields = self.cmaps

                    state.cmap_range_visible = True
                    self.reset_cmap_range(ctrl_mesh_name)
                    state.cmap_range = self.cmap_range[ctrl_mesh_name]
                    state.cmap_range_min = self.cmap_range[ctrl_mesh_name][0]
                    state.cmap_range_max = self.cmap_range[ctrl_mesh_name][1]

                elif state.active_var in self.categorical_vars[ctrl_mesh_name]:
                    state.cmap_visible = False
                    state.cmap_range_visible = False

            state.opacity = self.opacity[ctrl_mesh_name]

        @state.change("opacity")
        def update_opacity(opacity, **kwargs):
            name = state.ctrl_mesh_name
            self.opacity = (name, opacity)

        @state.change("active_var")
        def update_active_var(active_var, **kwargs):
            name = state.ctrl_mesh_name
            if name in self.interval_actor_names + self.point_actor_names:
                self.active_var = (name, active_var)
                if active_var in self.continuous_vars[name]:
                    state.cmap = self.cmap[name]

                    self.reset_cmap_range(name)
                    state.cmap_range = self.cmap_range[name]
                    state.cmap_range_min = self.cmap_range[name][0]
                    state.cmap_range_max = self.cmap_range[name][1]

                    state.cmap_visible = True
                    state.cmap_range_visible = True

                else:
                    state.cmap_visible = False
                    state.cmap_range_visible = False

        @state.change("cmap")
        def update_cmap(cmap, **kwargs):
            name = state.ctrl_mesh_name
            if name in self.interval_actor_names + self.point_actor_names:
                self.cmap = (name, cmap)

        @state.change("cmap_range")
        def update_cmap_range(cmap_range, **kwargs):
            name = state.ctrl_mesh_name
            if name in self.interval_actor_names + self.point_actor_names:
                self.cmap_range = (name, (cmap_range[0], cmap_range[1]))

    @DrillDownPlotter.opacity.setter
    def opacity(self, key_value_pair):
        super(DrillDownTramePlotter, DrillDownTramePlotter).opacity.fset(
            self, key_value_pair
        )
        name, opacity = key_value_pair
        if name == self.state.ctrl_mesh_name:
            self.state.opacity = opacity
            self.state.flush()

    @DrillDownPlotter.active_var.setter
    def active_var(self, key_value_pair):
        super(DrillDownTramePlotter, DrillDownTramePlotter).active_var.fset(
            self, key_value_pair
        )
        name, active_var = key_value_pair
        if name == self.state.ctrl_mesh_name:
            self.state.active_var = active_var
            self.state.flush()

    @DrillDownPlotter.cmap.setter
    def cmap(self, key_value_pair):
        super(DrillDownTramePlotter, DrillDownTramePlotter).cmap.fset(
            self, key_value_pair
        )
        name, cmap = key_value_pair
        if name == self.state.ctrl_mesh_name:
            self.state.cmap = cmap
            self.state.flush()

    @DrillDownPlotter.cmap_range.setter
    def cmap_range(self, key_value_pair):
        super(DrillDownTramePlotter, DrillDownTramePlotter).cmap_range.fset(
            self, key_value_pair
        )
        name, cmap_range = key_value_pair
        if name == self.state.ctrl_mesh_name:
            self.state.cmap_range = cmap_range
            self.state.flush()

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
        actor = super(DrillDownTramePlotter, self).add_mesh(
            mesh,
            name=name,
            opacity=opacity,
            pickable=pickable,
            filter_opacity=filter_opacity,
            selection_color=selection_color,
            accelerated_selection=accelerated_selection,
            *args,
            **kwargs,
        )
        if self._ui:
            if name not in self.state.ctrl_mesh_name_fields:
                if name in self.mesh_names:  # ignore selection or filter meshes
                    self.state.ctrl_mesh_name_fields = self.mesh_names.copy()
                    self.state.ctrl_mesh_name = name

                    if name in self.point_actor_names + self.interval_actor_names:
                        self.state.divider_visible = True
                        self.state.active_var_visible = True

                        if self.active_var[name] in self.continuous_vars[name]:
                            self.state.cmap_visible = True
                            self.state.cmap_range_visible = True

                        else:
                            self.state.cmap_visible = False
                            self.state.cmap_range_visible = False

                    else:
                        self.state.divider_visible = False
                        self.state.active_var_visible = False
                        self.state.cmap_visible = False
                        self.state.cmap_range_visible = False

                    self.state.flush()

        return actor


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

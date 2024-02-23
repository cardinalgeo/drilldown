from trame.widgets import vuetify, markdown

from trame.ui.vuetify import SinglePageWithDrawerLayout
from trame_server.utils.browser import open_browser

from pyvista.trame import PyVistaRemoteView
from pyvista.trame.jupyter import elegantly_launch

from ..plotter import Plotter
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


class DrillDownPlotter(Plotter):
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

            # update active array name
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

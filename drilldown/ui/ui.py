from trame.widgets import vuetify, markdown

from trame.ui.vuetify import SinglePageWithDrawerLayout
from trame_server.utils.browser import open_browser

from pyvista.trame import PyVistaRemoteView, get_viewer
from pyvista.trame.jupyter import elegantly_launch

from ..plotter import Plotter
from ..utils import is_jupyter
from ..layer.layer import IntervalDataLayer, PointDataLayer
from .layer_list import LayerListUI
from .controls import ControlsUI


class DrillDownPlotter(Plotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ui = None
        self.layer_list_ui = None
        self.state.active_layer_name = None

    def show(self, inline=False, menu_button=False, return_viewer=False):
        if return_viewer == True:
            viewer = super().show(return_viewer=True)

            return viewer

        self._ui = self._initialize_ui(menu_button=menu_button)
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

    def _initialize_ui(self, menu_button=False):
        with SinglePageWithDrawerLayout(self.server) as layout:
            layout.title.set_text("")
            with layout.footer as footer:
                footer.clear()
                footer.style = "z-index: 1000"
                vuetify.VCardText("DrillDown")
                # md = markdown.Markdown()
                # md.update("DrillDown")
            with layout.content:
                with vuetify.VContainer(
                    fluid=True,
                    classes="fill-height pa-0 ma-0",
                ):
                    if menu_button == True:
                        get_viewer(self, server=self.server).ui(
                            mode="server", collapse_menu=True
                        )
                    else:
                        PyVistaRemoteView(self)

            with layout.drawer as self.drawer:
                with vuetify.VContainer(
                    classes="fill-height pa-0 ma-0", style="overflow: hidden;"
                ) as self.drawer_content:
                    active_layer = self.layers[-1]
                    self.controls_ui = ControlsUI(active_layer)
                    self.layer_list_ui = LayerListUI()

        for layer in self.layers:
            self.layer_list_ui.add_layer(layer)

        layout.flush_content()
        self.state.flush()

        return layout

    def _initialize_engine(self):
        state = self.server.state

        @state.change("active_layer_name")
        def update_controls_mesh(active_layer_name, **kwargs):
            layer = self.layers[active_layer_name]

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
            layer = self.layers[state.active_layer_name]

            if visibility != layer.visibility:
                layer.visibility = visibility

        @state.change("opacity")
        def update_opacity(opacity, **kwargs):
            layer = self.layers[state.active_layer_name]

            if opacity != layer.opacity:
                layer.opacity = opacity

        @state.change("active_array_name")
        def update_active_array_name(active_array_name, **kwargs):
            layer = self.layers[state.active_layer_name]

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
            layer = self.layers[state.active_layer_name]

            if cmap != layer.cmap:
                layer.cmap = cmap

        @state.change("clim")
        def update_clim(clim, **kwargs):
            layer = self.layers[state.active_layer_name]
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

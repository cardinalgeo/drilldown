from trame.app import get_server
from trame.widgets import vuetify, plotly
from pyvista.trame import PyVistaRemoteView
from pyvista.trame.jupyter import elegantly_launch
from trame_server.utils.browser import open_browser
from trame.ui.vuetify import SinglePageLayout

from plotly import express as px
import numpy as np
import uuid


from ..utils import convert_to_numpy_array, is_jupyter


class PlotlyPlot:
    def __init__(self, name=None, server=None, *args, **kwargs):
        if name is not None:
            self.name = name
        else:
            self.name = str(uuid.uuid4())

        if server is not None:
            self.server = server
        else:
            self.server = get_server(self.name)

        self.state = self.server.state
        self.ctrl = self.server.controller
        self.server.client_type = "vue2"

        self._ui = self._initialize_ui()

        self.data = None
        self.ids = None
        self.selected_ids = None

        self.layer = None

    def show(self, inline=True):
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
        with SinglePageLayout(self.server, template_name=self.name) as layout:
            layout.toolbar.hide()
            layout.footer.hide()
            with layout.content:
                with vuetify.VContainer(
                    fluid=True,
                    classes="fill-height pa-0 ma-0",
                ):
                    html_plot = plotly.Figure(
                        selected=(
                            self._on_plot_selection,
                            "[$event?.points.map(({pointIndex}) => pointIndex)]",
                        ),
                        # click=(
                        #     self.ctrl.on_plotly_plot_click,
                        #     "[$event?.points.map(({pointIndex}) => pointIndex)]",
                        # ),
                        deselect=(
                            self._on_plot_deselection,
                            "[$event]",
                        ),
                    )
                self.ctrl.plotly_plot_view_update = html_plot.update

        return layout

    def _on_plot_selection(self, ids):
        if ids is not None:
            self.selected_ids = ids

            if self.layer is not None:
                self.layer.selected_ids = np.array(self.ids)[ids]

    def _on_plot_deselection(self, event):
        self.selected_ids = None

        if self.layer is not None:
            self.layer.selected_ids = np.array(self.ids)

    def _on_scatter_plot_click(self, ids):
        pass

    def reset_resolution(self, app):
        app.reset_resolution()


class ScatterPlot(PlotlyPlot):
    def __init__(self, data, x, y, **kwargs):
        super().__init__()
        self.data = data
        fig = px.scatter(data, x=x, y=y, **kwargs)
        self.fig = fig
        self.ctrl.plotly_plot_view_update(fig)


class Scatter3dPlot(PlotlyPlot):
    def __init__(self, data, x, y, z, **kwargs):
        super().__init__()
        self.data = data
        fig = px.scatter_3d(data, x=x, y=y, z=z, **kwargs)
        self.fig = fig
        self.ctrl.plotly_plot_view_update(fig)


class ScatterTernaryPlot(PlotlyPlot):
    def __init__(self, data, a, b, c, **kwargs):
        server = kwargs.get("server", None)
        super().__init__(server)
        kwargs.pop("server", None)

        self.data = data
        fig = px.scatter_ternary(data, a=a, b=b, c=c, **kwargs)
        self.fig = fig
        self.ctrl.plotly_plot_view_update(fig)


class ScatterDimensionsPlot(PlotlyPlot):
    def __init__(self, data, dimensions, **kwargs):
        server = kwargs.get("server", None)
        super().__init__(server)
        kwargs.pop("server", None)

        self.data = data
        fig = px.scatter_matrix(data, dimensions=dimensions, **kwargs)
        self.fig = fig
        self.ctrl.plotly_plot_view_update(fig)


class BarPlot(PlotlyPlot):
    def __init__(self, data, x, y, **kwargs):
        super().__init__()
        self.data = data
        fig = px.bar(data, x=x, y=y, **kwargs)
        self.fig = fig
        self.ctrl.plotly_plot_view_update(fig)


class Histogram(PlotlyPlot):
    def __init__(self, data, x, **kwargs):
        super().__init__()
        self.data = data
        fig = px.histogram(data, x=x, **kwargs)
        self.fig = fig
        self.ctrl.plotly_plot_view_update(fig)

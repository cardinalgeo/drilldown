from trame.app import get_server
from trame.widgets import vuetify, plotly
from pyvista.trame import PyVistaRemoteView
from pyvista.trame.jupyter import elegantly_launch
from trame_server.utils.browser import open_browser
from trame.ui.vuetify import SinglePageLayout

from plotly import express as px


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


class PlotlyPlot:
    def __init__(self, *args, **kwargs):
        self.server = get_server(name="plotly")
        self.state = self.server.state
        self.ctrl = self.server.controller
        self.server.client_type = "vue2"

        self._ui = self._initialize_ui()
        self._initialize_engine()

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
        with SinglePageLayout(self.server) as layout:
            layout.toolbar.hide()
            layout.footer.hide()
            with layout.content:
                with vuetify.VContainer(
                    fluid=True,
                    classes="fill-height pa-0 ma-0",
                ):
                    html_plot = plotly.Figure(
                        selected=(
                            self.ctrl.on_plotly_plot_selection,
                            "[$event?.points.map(({pointIndex}) => pointIndex)]",
                        ),
                        click=(
                            self.ctrl.on_plotly_plot_click,
                            "[$event?.points.map(({pointIndex}) => pointIndex)]",
                        ),
                    )
                self.ctrl.plotly_plot_view_update = html_plot.update

        return layout

    def _initialize_engine(self):
        self.ctrl.on_scatter_plot_selection = self._on_scatter_plot_selection
        self.ctrl.on_scatter_plot_click = self._on_scatter_plot_click

    def _on_scatter_plot_selection(self, ids):
        pass

    def _on_scatter_plot_click(self, ids):
        pass


class ScatterPlot(PlotlyPlot):
    def __init__(self, data, x, y, **kwargs):
        super().__init__()
        fig = px.scatter(data, x=x, y=y, **kwargs)
        self.fig = fig
        self.ctrl.plotly_plot_view_update(fig)

    # def update_selection(self, selected_idx):
    #     self.fig.data[0].update(
    #         selectedpoints=selected_idx,
    #         selected={"marker": {"color": "red"}},
    #         unselected={"marker": {"opacity": 0.5}},
    #     )
    #     return self.fig


def on_scatter_plot_selection(self, ids):
    selected_interval_cells = []
    if ids:
        holes = scene.filters["drillhole_intervals"]
        for id in ids:
            selected_interval_cells += np.arange(
                self.cells_per_interval * id,
                self.cells_per_interval * id + self.cells_per_interval,
            ).tolist()
    state.selected_interval_cells = selected_interval_cells
    state.selected_intervals = ids

    ctrl.update_deposit_viewer_selection("interval", state.selected_interval_cells)

    ctrl.deposit_viewer_view_update()


def on_scatter_plot_click(self, ids):
    self.on_scatter_plot_selection(ids)

    scatter = ctrl.get_scatter_plot()
    scatter.update_selection(state.selected_intervals)
    ctrl.scatter_plot_view_update(scatter.fig)


def on_scatter_plot_relayout(self, event):
    state.scatter_x_range = {state.scatter_x: None}
    state.scatter_y_range = {state.scatter_y: None}
    for key in event.keys():
        if "autorange" in key:
            return
        elif "xaxis" in key:
            state.scatter_x_range = {
                state.scatter_x: [event["xaxis.range[0]"], event["xaxis.range[1]"]]
            }

        elif "yaxis" in key:
            state.scatter_y_range = {
                state.scatter_y: [event["yaxis.range[0]"], event["yaxis.range[1]"]]
            }

    return

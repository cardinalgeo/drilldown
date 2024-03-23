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
    """Base class for Plotly plots.

    Provides functionality for connecting a plot to a layer (and the associated plotter) for cross-filtering.

    Parameters
    ----------
    name : str, optional
        Name of the plot. If not provided, a random name will be generated.

    server : trame_server.core.Server, optional
        Server to use. If not provided, a new server will be created.

    """

    def __init__(self, name=None, server=None):
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

        self._layer = None

    def show(self, inline=True):
        """Show the plot.

        Starts the server of a trame microapp and displays the plot.

        Parameters
        ----------
        inline : bool, optional
            Whether to show the plot inline in a Jupyter notebook. If False, the plot will be displayed in a new browser window. Default is True.

        """
        self.update()
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
        """Initialize the UI for the plot.

        Returns
        -------
        trame.ui.vuetify.SinglePageLayout
            The UI for the plot.

        """
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

    @property
    def layer(self):
        """Layer connected with the plot."""
        return self._layer

    @property
    def layer_relationship(self):
        """Relationship between the plot and the connected layer.

        "selected" means the plot is populated by the layer's selected data; "filtered" means the plot is populated by the layer's filtered data; None means the plot is populated by the layer's full data

        Returns
        -------
        str
            Relationship between the plot and the connected layer. Can be 'selected', 'filtered', or None.
        """
        return self._layer_relationship

    def connect_layer(self, layer, relationship=None):
        """Connect the plot to a layer.

        Parameters
        ----------
        layer : drilldown.layer.PointDataLayer or drilldown.layer.IntervalDataLayer
            Layer to connect to.

        relationship : str, optional
            Relationship between the plot and the layer. Can be 'selected', 'filtered', or None. Default is None.

        """
        if relationship is not None:
            if relationship not in ["selected", "filtered"]:
                raise ValueError(
                    "Relationship must be 'selected', 'filtered', or None."
                )

        self._layer = layer
        self._layer.plot = self
        self._layer_relationship = relationship

    def _on_plot_selection(self, ids):
        """Callback for when points are selected on the plot.

        Updates the selected data for the connected layer.

        Parameters
        ----------
        ids : list
            List of selected data indices.

        """
        if ids is not None:
            self.selected_ids = ids

            if self.layer is not None:
                self.layer.selected_ids = np.array(self.ids)[ids]

    def _on_plot_deselection(self, event):
        """Callback for when points are deselected on the plot.

        If the layer relationship is 'selected', restores the layer selection. Otherwise, resets (i.e., removes) the layer selection.

        Parameters
        ----------
        event : dict
            Event data. Unused

        """
        self.selected_ids = None

        if self.layer is not None:
            if self.layer_relationship == "selected":
                self.layer.selected_ids = np.array(self.ids)
            else:
                self.layer._reset_selection()

    def _on_scatter_plot_click(self, ids):
        """Callback for when points are clicked on the plot.

        Parameters
        ----------
        ids : list
            List of clicked data indices.

        """
        pass

    def update(self, *args, **kwargs):
        """Update the plot."""
        self.ctrl.plotly_plot_view_update(*args, **kwargs)


class ScatterPlot(PlotlyPlot):
    """2D scatter plot using Plotly Express syntax.

    Stores a plotly.graph_objects.Figure object using the `fig` attribute for further modification.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.

    x : str
        Name of the column to use for the x-axis.

    y : str
        Name of the column to use for the y-axis.

    """

    def __init__(self, data, x, y, **kwargs):
        super().__init__()
        self.data = data
        fig = px.scatter(data, x=x, y=y, **kwargs)
        self.fig = fig
        self.update(fig)


class Scatter3dPlot(PlotlyPlot):
    """3D scatter plot using Plotly Express syntax.

    Stores a plotly.graph_objects.Figure object using the `fig` attribute for further modification.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.

    x : str
        Name of the column to use for the x-axis.

    y : str
        Name of the column to use for the y-axis.

    z : str
        Name of the column to use for the z-axis.

    """

    def __init__(self, data, x, y, z, **kwargs):
        super().__init__()
        self.data = data
        fig = px.scatter_3d(data, x=x, y=y, z=z, **kwargs)
        self.fig = fig
        self.update(fig)


class ScatterTernaryPlot(PlotlyPlot):
    """Ternary scatter plot using Plotly Express syntax.

    Stores a plotly.graph_objects.Figure object using the `fig` attribute for further modification.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.

    a : str
        Name of the column to use for the a-axis in ternary coordinates.

    b : str
        Name of the column to use for the b-axis in ternary coordinates.

    c : str
        Name of the column to use for the c-axis in ternary coordinates.

    """

    def __init__(self, data, a, b, c, **kwargs):
        server = kwargs.get("server", None)
        super().__init__(server)
        kwargs.pop("server", None)

        self.data = data
        fig = px.scatter_ternary(data, a=a, b=b, c=c, **kwargs)
        self.fig = fig
        self.update(fig)


class ScatterDimensionsPlot(PlotlyPlot):
    """Scatter matrix plot using Plotly Express syntax.

    Stores a plotly.graph_objects.Figure object using the `fig` attribute for further modification.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.

    dimensions : list
        List of column names to use for each subplot in the matrix.

    """

    def __init__(self, data, dimensions, **kwargs):
        server = kwargs.get("server", None)
        super().__init__(server)
        kwargs.pop("server", None)

        self.data = data
        fig = px.scatter_matrix(data, dimensions=dimensions, **kwargs)
        self.fig = fig
        self.update(fig)


class BarPlot(PlotlyPlot):
    """Bar plot using Plotly Express syntax.

    Stores a plotly.graph_objects.Figure object using the `fig` attribute for further modification.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.

    x : str
        Name of the column to use for the x-axis.

    y : str
        Name of the column to use for the y-axis.

    """

    def __init__(self, data, x, y, **kwargs):
        super().__init__()
        self.data = data
        fig = px.bar(data, x=x, y=y, **kwargs)
        self.fig = fig
        self.update(fig)


class Histogram(PlotlyPlot):
    """Histogram using Plotly Express syntax.

    Stores a plotly.graph_objects.Figure object using the `fig` attribute for further modification.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to plot.

    x : str
        Name of the column to use for the x-axis.

    """

    def __init__(self, data, x, **kwargs):
        super().__init__()
        self.data = data
        fig = px.histogram(data, x=x, **kwargs)
        self.fig = fig
        self.update(fig)

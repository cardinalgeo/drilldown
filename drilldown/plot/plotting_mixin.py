class Plotting2dMixin:
    """Mixin class for [mostly] 2D plotting methods using Plotly Express syntax."""

    def __init__(self):
        pass

    def scatter_plot(self, x, y, **kwargs):
        """Create a scatter plot with all of the layer's data using Plotly Express syntax.

        Parameters
        ----------
        x : str
            The array name for the x-axis data.

        y : str
            The array name for the y-axis data.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.ScatterPlot
            The figure object.

        """
        from .plotly_plots import ScatterPlot

        fig = ScatterPlot(self.data, x, y, **kwargs)

        fig.connect_layer(self)
        fig.ids = self.ids

        return fig

    def scatter_plot_from_selection(self, x, y, **kwargs):
        """Create a scatter plot with the layer's selected data using Plotly Express syntax.

        Parameters
        ----------
        x : str
            The array name for the x-axis data.

        y : str
            The array name for the y-axis data.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.ScatterPlot
            The figure object.

        """
        from .plotly_plots import ScatterPlot

        fig = ScatterPlot(self.selected_data, x, y, **kwargs)

        fig.connect_layer(self, relationship="selected")
        fig.ids = self.selected_ids

        return fig

    def scatter_plot_from_filter(self, x, y, **kwargs):
        """Create a scatter plot with the layer's "filtered-in" data using Plotly Express syntax.

        Parameters
        ----------
        x : str
            The array name for the x-axis data.

        y : str
            The array name for the y-axis data.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.ScatterPlot
            The figure object.

        """
        from .plotly_plots import ScatterPlot

        fig = ScatterPlot(self.filtered_data, x, y, **kwargs)

        fig.connect_layer(self, relationship="filtered")
        fig.ids = self.filtered_ids

        return fig

    def scatter_3d_plot(self, x, y, z, **kwargs):
        """Create a 3D scatter plot with all of the layer's data using Plotly Express syntax.

        Parameters
        ----------
        x : str
            The array name for the x-axis data.

        y : str
            The array name for the y-axis data.

        z : str
            The array name for the z-axis data.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.Scatter3dPlot
            The figure object.

        """
        from .plotly_plots import Scatter3dPlot

        fig = Scatter3dPlot(self.data, x, y, z, **kwargs)

        fig.connect_layer(self)
        fig.ids = self.ids

        return fig

    def scatter_3d_plot_from_selection(self, x, y, z, **kwargs):
        """Create a 3D scatter plot with the layer's selected data using Plotly Express syntax.

        Parameters
        ----------
        x : str
            The array name for the x-axis data.

        y : str
            The array name for the y-axis data.

        z : str
            The array name for the z-axis data.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.Scatter3dPlot
            The figure object.

        """
        from .plotly_plots import Scatter3dPlot

        fig = Scatter3dPlot(self.selected_data, x, y, z, **kwargs)

        fig.connect_layer(self, relationship="selected")
        fig.ids = self.selected_ids

        return fig

    def scatter_3d_plot_from_filter(self, x, y, z, **kwargs):
        """Create a 3D scatter plot with the layer's "filtered-in" data using Plotly Express syntax.

        Parameters
        ----------
        x : str
            The array name for the x-axis data.

        y : str
            The array name for the y-axis data.

        z : str
            The array name for the z-axis data.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.Scatter3dPlot
            The figure object.

        """
        from .plotly_plots import Scatter3dPlot

        fig = Scatter3dPlot(self.filtered_data, x, y, z, **kwargs)

        fig.connect_layer(self, relationship="filtered")
        fig.ids = self.filtered_ids

        return fig

    def scatter_ternary_plot(self, a, b, c, **kwargs):
        """Create a ternary scatter plot with all of the layer's data using Plotly Express syntax.

        Parameters
        ----------
        a : str
            The array name for the a-axis data.

        b : str
            The array name for the b-axis data.

        c : str
            The array name for the c-axis data.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.ScatterTernaryPlot
            The figure object.

        """
        from .plotly_plots import ScatterTernaryPlot

        fig = ScatterTernaryPlot(self.data, a, b, c, **kwargs)

        fig.connect_layer(self)
        fig.ids = self.ids

        return fig

    def scatter_ternary_plot_from_selection(self, a, b, c, **kwargs):
        """Create a ternary scatter plot with the layer's selected data using Plotly Express syntax.

        Parameters
        ----------
        a : str
            The array name for the a-axis data.

        b : str
            The array name for the b-axis data.

        c : str
            The array name for the c-axis data.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.ScatterTernaryPlot
            The figure object.

        """
        from .plotly_plots import ScatterTernaryPlot

        fig = ScatterTernaryPlot(self.selected_data, a, b, c, **kwargs)

        fig.plotter = self

        fig.connect_layer(self, relationship="selected")
        fig.ids = self.selected_ids

        return fig

    def scatter_ternary_plot_from_filter(self, a, b, c, **kwargs):
        """Create a ternary scatter plot with the layer's "filtered-in" data using Plotly Express syntax.

        Parameters
        ----------
        a : str
            The array name for the a-axis data.

        b : str
            The array name for the b-axis data.

        c : str
            The array name for the c-axis data.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.ScatterTernaryPlot
            The figure object.

        """
        from .plotly_plots import ScatterTernaryPlot

        fig = ScatterTernaryPlot(self.filtered_data, a, b, c, **kwargs)

        fig.plotter = self

        fig.connect_layer(self, relationship="filtered")
        fig.ids = self.filtered_ids

        return fig

    def scatter_dimensions_plot(self, dimensions, **kwargs):
        """Create a scatter plot matrix (or SPLOM) with all of the layer's data using Plotly Express syntax.

        Parameters
        ----------
        dimensions : list
            A list of array names to be used for each subplot in the matrix.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.ScatterDimensionsPlot
            The figure object.

        """
        from .plotly_plots import ScatterDimensionsPlot

        fig = ScatterDimensionsPlot(self.data, dimensions, **kwargs)

        fig.connect_layer(self)
        fig.ids = self.ids

        return fig

    def scatter_dimensions_plot_from_selection(self, dimensions, **kwargs):
        """Create a scatter plot matrix (or SPLOM) with the layer's selected data using Plotly Express syntax.

        Parameters
        ----------
        dimensions : list
            A list of array names to be used for each subplot in the matrix.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.ScatterDimensionsPlot
            The figure object.

        """
        from .plotly_plots import ScatterDimensionsPlot

        fig = ScatterDimensionsPlot(self.selected_data, dimensions, **kwargs)

        fig.plotter = self

        fig.connect_layer(self, relationship="selected")
        fig.ids = self.selected_ids

        return fig

    def scatter_dimensions_plot_from_filter(self, dimensions, **kwargs):
        """Create a scatter plot matrix (or SPLOM) with the layer's "filtered-in" data using Plotly Express syntax.

        Parameters
        ----------
        dimensions : list
            A list of array names to be used for each subplot in the matrix.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.ScatterDimensionsPlot
            The figure object.

        """
        from .plotly_plots import ScatterDimensionsPlot

        fig = ScatterDimensionsPlot(self.filtered_data, dimensions, **kwargs)

        fig.plotter = self

        fig.connect_layer(self, relationship="filtered")
        fig.ids = self.filtered_ids

        return fig

    def bar_plot(self, x, y, **kwargs):
        """Create a bar plot with all of the layer's data using Plotly Express syntax.

        Parameters
        ----------
        x : str
            The array name for the x-axis data.

        y : str
            The array name for the y-axis data.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.BarPlot
            The figure object.

        """
        from .plotly_plots import BarPlot

        fig = BarPlot(self.data, x, y, **kwargs)

        fig.connect_layer(self)
        fig.ids = self.ids

        return fig

    def bar_plot_from_selection(self, x, y, **kwargs):
        """Create a bar plot with the layer's selected data using Plotly Express syntax.

        Parameters
        ----------
        x : str
            The array name for the x-axis data.

        y : str
            The array name for the y-axis data.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.BarPlot
            The figure object.

        """
        from .plotly_plots import BarPlot

        fig = BarPlot(self.selected_data, x, y, **kwargs)

        fig.plotter = self

        fig.connect_layer(self, relationship="selected")
        fig.ids = self.selected_ids

        return fig

    def bar_plot_from_filter(self, x, y, **kwargs):
        """Create a bar plot with the layer's "filtered-in" data using Plotly Express syntax.

        Parameters
        ----------
        x : str
            The array name for the x-axis data.

        y : str
            The array name for the y-axis data.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.BarPlot
            The figure object.

        """
        from .plotly_plots import BarPlot

        fig = BarPlot(self.filtered_data, x, y, **kwargs)

        fig.plotter = self

        fig.connect_layer(self, relationship="filtered")
        fig.ids = self.filtered_ids

        return fig

    def histogram(self, x, **kwargs):
        """Create a histogram with all of the layer's data using Plotly Express syntax.

        Parameters
        ----------
        x : str
            The array name for the x-axis data.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.Histogram
            The figure object.

        """
        from .plotly_plots import Histogram

        fig = Histogram(self.data, x, **kwargs)

        fig.connect_layer(self)
        fig.ids = self.ids

        return fig

    def histogram_from_selection(self, x, **kwargs):
        """Create a histogram with the layer's selected data using Plotly Express syntax.

        Parameters
        ----------
        x : str
            The array name for the x-axis data.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.Histogram
            The figure object.

        """
        from .plotly_plots import Histogram

        fig = Histogram(self.selected_data, x, **kwargs)

        fig.plotter = self

        fig.connect_layer(self, relationship="selected")
        fig.ids = self.selected_ids

        return fig

    def histogram_from_filter(self, x, **kwargs):
        """Create a histogram with the layer's "filtered-in" data using Plotly Express syntax.

        Parameters
        ----------
        x : str
            The array name for the x-axis data.

        Returns
        ------
        fig : drilldown.plot.plotly_plots.Histogram
            The figure object.

        """
        from .plotly_plots import Histogram

        fig = Histogram(self.filtered_data, x, **kwargs)

        fig.plotter = self

        fig.connect_layer(self, relationship="filtered")
        fig.ids = self.filtered_ids

        return fig

class Plotting2dMixin:
    def __init__(self):
        pass

    def scatter_plot(self, x, y, **kwargs):
        from .plotly_plots import ScatterPlot

        fig = ScatterPlot(self.all_data, x, y, **kwargs)

        fig.layer = self
        fig.ids = self.all_ids

        return fig

    def selected_scatter_plot(self, x, y, **kwargs):
        from .plotly_plots import ScatterPlot

        fig = ScatterPlot(self.selected_data, x, y, **kwargs)

        fig.layer = self
        fig.ids = self.selected_ids

        return fig

    def filtered_scatter_plot(self, x, y, **kwargs):
        from .plotly_plots import ScatterPlot

        fig = ScatterPlot(self.filtered_data, x, y, **kwargs)

        fig.layer = self
        fig.ids = self.filtered_ids

        return fig

    def scatter_3d_plot(self, x, y, z, **kwargs):
        from .plotly_plots import Scatter3dPlot

        fig = Scatter3dPlot(self.all_data, x, y, z, **kwargs)

        fig.layer = self
        fig.ids = self.all_ids

        return fig

    def selected_scatter_3d_plot(self, x, y, z, **kwargs):
        from .plotly_plots import Scatter3dPlot

        fig = Scatter3dPlot(self.selected_data, x, y, z, **kwargs)

        fig.layer = self
        fig.ids = self.selected_ids

        return fig

    def filtered_scatter_3d_plot(self, x, y, z, **kwargs):
        from .plotly_plots import Scatter3dPlot

        fig = Scatter3dPlot(self.filtered_data, x, y, z, **kwargs)

        fig.layer = self
        fig.ids = self.filtered_ids

        return fig

    def scatter_ternary_plot(self, a, b, c, **kwargs):
        from .plotly_plots import ScatterTernaryPlot

        fig = ScatterTernaryPlot(self.all_data, a, b, c, **kwargs)

        fig.layer = self
        fig.ids = self.all_ids

        return fig

    def selected_scatter_ternary_plot(self, a, b, c, **kwargs):
        from .plotly_plots import ScatterTernaryPlot

        fig = ScatterTernaryPlot(self.selected_data, a, b, c, **kwargs)

        fig.plotter = self

        fig.layer = self
        fig.ids = self.selected_ids

        return fig

    def filtered_scatter_ternary_plot(self, a, b, c, **kwargs):
        from .plotly_plots import ScatterTernaryPlot

        fig = ScatterTernaryPlot(self.filtered_data, a, b, c, **kwargs)

        fig.plotter = self

        fig.layer = self
        fig.ids = self.filtered_ids

        return fig

    def scatter_dimensions_plot(self, dimensions, **kwargs):
        from .plotly_plots import ScatterDimensionsPlot

        fig = ScatterDimensionsPlot(self.all_data, dimensions, **kwargs)

        fig.layer = self
        fig.ids = self.all_ids

        return fig

    def selected_scatter_dimensions_plot(self, dimensions, **kwargs):
        from .plotly_plots import ScatterDimensionsPlot

        fig = ScatterDimensionsPlot(self.selected_data, dimensions, **kwargs)

        fig.plotter = self

        fig.layer = self
        fig.ids = self.selected_ids

        return fig

    def filtered_scatter_dimensions_plot(self, dimensions, **kwargs):
        from .plotly_plots import ScatterDimensionsPlot

        fig = ScatterDimensionsPlot(self.filtered_data, dimensions, **kwargs)

        fig.plotter = self

        fig.layer = self
        fig.ids = self.filtered_ids

        return fig

    def bar_plot(self, x, y, **kwargs):
        from .plotly_plots import BarPlot

        fig = BarPlot(self.all_data, x, y, **kwargs)

        fig.layer = self
        fig.ids = self.all_ids

        return fig

    def selected_bar_plot(self, x, y, **kwargs):
        from .plotly_plots import BarPlot

        fig = BarPlot(self.selected_data, x, y, **kwargs)

        fig.plotter = self

        fig.layer = self
        fig.ids = self.selected_ids

        return fig

    def filtered_bar_plot(self, x, y, **kwargs):
        from .plotly_plots import BarPlot

        fig = BarPlot(self.filtered_data, x, y, **kwargs)

        fig.plotter = self

        fig.layer = self
        fig.ids = self.filtered_ids

        return fig

    def histogram(self, x, **kwargs):
        from .plotly_plots import Histogram

        fig = Histogram(self.all_data, x, **kwargs)

        fig.layer = self
        fig.ids = self.all_ids

        return fig

    def selected_histogram(self, x, **kwargs):
        from .plotly_plots import Histogram

        fig = Histogram(self.selected_data, x, **kwargs)

        fig.plotter = self

        fig.layer = self
        fig.ids = self.selected_ids

        return fig

    def filtered_histogram(self, x, **kwargs):
        from .plotly_plots import Histogram

        fig = Histogram(self.filtered_data, x, **kwargs)

        fig.plotter = self

        fig.layer = self
        fig.ids = self.filtered_ids

        return fig

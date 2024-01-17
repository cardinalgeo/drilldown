class Plotting2dMixin:
    def __init__(self):
        pass

    def selected_scatter_plot(self, x, y, **kwargs):
        from .plotly_plots import ScatterPlot

        fig = ScatterPlot(self.selected_data(), x, y, **kwargs)

        fig.plotter = self
        fig.actor_name = self.selection_actor_name.split(" ")[0]
        if fig.actor_name in self.interval_actor_names:
            fig.ids = self.selected_intervals
        elif fig.actor_name in self.point_actor_names:
            fig.ids = self.selected_points

        return fig

    def selected_scatter_ternary_plot(self, a, b, c, **kwargs):
        from .plotly_plots import ScatterTernaryPlot

        fig = ScatterTernaryPlot(self.selected_data(), a, b, c, **kwargs)

        fig.plotter = self
        fig.actor_name = self.selection_actor_name.split(" ")[0]
        if fig.actor_name in self.interval_actor_names:
            fig.ids = self.selected_intervals
        elif fig.actor_name in self.point_actor_names:
            fig.ids = self.selected_points

        return fig

    def selected_scatter_dimensions_plot(self, dimensions, **kwargs):
        from .plotly_plots import ScatterDimensionsPlot

        fig = ScatterDimensionsPlot(self.selected_data(), dimensions, **kwargs)

        fig.plotter = self
        fig.actor_name = self.selection_actor_name.split(" ")[0]
        if fig.actor_name in self.interval_actor_names:
            fig.ids = self.selected_intervals
        elif fig.actor_name in self.point_actor_names:
            fig.ids = self.selected_points

        return fig

    def selected_bar_plot(self, x, y, **kwargs):
        from .plotly_plots import BarPlot

        fig = BarPlot(self.selected_data(), x, y, **kwargs)

        fig.plotter = self
        fig.actor_name = self.selection_actor_name.split(" ")[0]
        if fig.actor_name in self.interval_actor_names:
            fig.ids = self.selected_intervals
        elif fig.actor_name in self.point_actor_names:
            fig.ids = self.selected_points

        return fig

    def selected_histogram(self, x, **kwargs):
        from .plotly_plots import Histogram

        fig = Histogram(self.selected_data(), x, **kwargs)

        fig.plotter = self
        fig.actor_name = self.selection_actor_name.split(" ")[0]
        if fig.actor_name in self.interval_actor_names:
            fig.ids = self.selected_intervals
        elif fig.actor_name in self.point_actor_names:
            fig.ids = self.selected_points

        return fig

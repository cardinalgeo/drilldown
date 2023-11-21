from plotly.subplots import make_subplots
from plotly import graph_objects as go
import numpy as np


def interleave_intervals(depths, values=None, connected=True):
    if connected == True:
        # construct new depths
        depths_new = np.empty(depths.shape[0] * 2)
        depths_new[0::2] = depths[:, 0]
        depths_new[1::2] = depths[:, 1]

        if values is not None:
            # construct new values
            values_new = np.empty(values.shape[0] * 2)
            values_new[0::2] = values
            values_new[1::2] = values

            return depths_new, values_new

        else:
            return depths_new


class DrillLog:
    def __init__(self):
        # set widths of columns by type and initialize list of widths
        self.width_categorical_interval = 0.1
        self.width_continuous_interval = 0.3
        self.width_categorical_point = 0.05
        self.width_continuous_point = 0.03
        self.col_widths = []

        # set number of rows
        self.n_rows = 1

        # set collective title and subplot titles
        self.title = None
        self.subplot_titles = []

        # initialize data
        self.categorical_interval_data = {}
        self.continuous_interval_data = {}
        self.categorical_point_data = {}
        self.continuous_point_data = {}

        # initialize depth range
        self.depth_range = [-np.inf, np.inf]

    def create_figure(self, y_axis_label=None, title=None):
        self.n_categorical_interval_cols = len(self.categorical_interval_data.keys())
        self.n_continuous_interval_cols = len(self.continuous_interval_data.keys())
        self.n_categorical_point_cols = len(self.categorical_point_data.keys())
        self.n_continuous_point_cols = len(self.continuous_point_data.keys())
        self.n_cols = (
            self.n_categorical_interval_cols
            + self.n_continuous_interval_cols
            + self.n_categorical_point_cols
            + self.n_continuous_point_cols
        )
        self._update_col_widths()

        fig = make_subplots(
            rows=self.n_rows,
            cols=self.n_cols,
            shared_yaxes=True,
            column_widths=self.col_widths,
            subplot_titles=self.subplot_titles,
        )
        self.fig = fig

        if self.continuous_interval_data:
            for col, var in enumerate(self.continuous_interval_data.keys()):
                name = var
                depths = self.continuous_interval_data[var]["depths"]
                values = self.continuous_interval_data[var]["values"]
                self._add_continuous_interval_data(
                    depths, values, name=name, col=col + 1
                )

        if self.categorical_interval_data:
            for var in self.categorical_interval_data.keys():
                pass

        if self.continuous_point_data:
            for var in self.continuous_point_data.keys():
                pass

        if self.categorical_point_data:
            for var in self.categorical_point_data.keys():
                pass

        # set moving horizontal line
        fig.update_yaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikecolor="#000000",
        )
        fig.update_traces(yaxis="y1")

        # set y axis label
        fig.update_yaxes(title_text=y_axis_label, row=1, col=1)

        # reverse y axis and set range
        fig.update_yaxes(range=[self.depth_range[1], self.depth_range[0]])

        # set subplot title font size
        fig.update_annotations(font_size=10)

        # set title, set font color for hover label
        fig.update_layout(
            title_text=title,
            margin=dict(l=75, r=20, t=40, b=20),
            hoverlabel=dict(font=dict(color="#000000")),
        )

    def add_categorical_interval_data(self, depths, values, name=None):
        self.categorical_interval_data[name] = {"depths": depths, "values": values}
        self.subplot_titles += [name]
        self._update_depth_range(depths)

    def add_continuous_interval_data(self, depths, values, name=None):
        self.continuous_interval_data[name] = {"depths": depths, "values": values}
        self.subplot_titles += [name]
        self._update_depth_range(depths)

    def add_categorical_point_data(self, depths, values, name=None):
        self.categorical_point_data[name] = {"depths": depths, "values": values}
        self.subplot_titles += [name]
        self._update_depth_range(depths)

    def add_continuous_point_data(self, depths, values, name=None):
        self.continuous_point_data[name] = {"depths": depths, "values": values}
        self.subplot_titles += [name]
        self._update_depth_range(depths)

    def _add_categorical_interval_data(self, name=None):
        pass

    def _add_continuous_interval_data(self, depths, values, name=None, col=None):
        add_col = False
        if add_col == True:
            # update subplot layout
            self.n_continuous_interval_cols += 1
            self._update_col_widths()

            # update subplot titles
            self.subplot_titles += [name]

            # create new figure
            self._create_figure()

        # duplicate depths and values for plotting
        depths, values = interleave_intervals(depths, values)

        # add data
        self.fig.add_trace(
            go.Scattergl(
                x=values,
                y=depths,
                showlegend=False,
                name=name,
                hoverinfo="x+name",
                mode="lines+markers",
                marker=dict(opacity=0),
            ),
            row=self.n_rows,
            col=col,
        )

    def _add_categorical_point_data(self, name=None):
        pass

    def _add_continuous_point_data(self, name=None):
        pass

    def _update_col_widths(self):
        self.col_widths = []
        self.col_widths += self.n_categorical_interval_cols * [
            self.width_categorical_interval
        ]
        self.col_widths += self.n_continuous_interval_cols * [
            self.width_continuous_interval
        ]
        self.col_widths += self.n_categorical_point_cols * [
            self.width_categorical_point
        ]
        self.col_widths += self.n_continuous_point_cols * [self.width_continuous_point]

    def _update_depth_range(self, depths):
        if depths.min() > self.depth_range[0]:
            self.depth_range[0] = depths.min()
        if depths.max() < self.depth_range[1]:
            self.depth_range[1] = depths.max()

    def show(self):
        return self.fig.show()

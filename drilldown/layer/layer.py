from vtk import (
    vtkCellPicker,
    vtkPointPicker,
    vtkCellLocator,
    vtkPointLocator,
    vtkHardwareSelector,
    vtkDataObject,
)
import pyvista as pv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..utils import convert_to_numpy_array
from ..image.image_mixin import ImageMixin
from ..plot.plotting_mixin import Plotting2dMixin
from ..drill_log import DrillLog


class _BaseLayer:
    def __init__(
        self,
        name,
        mesh,
        actor,
        plotter,
        visibility=True,
        opacity=1,
        selection_color="magenta",
        rel_selection_opacity=1,
        rel_filter_opacity=0.1,
    ):
        self.plotter = plotter
        self.state = self.plotter.state

        self.name = name
        self.state.layer_names = self.state.layer_names + [name]
        self.mesh = mesh
        self.actor = actor

        self._visibility = visibility
        self.state.visibility = visibility

        self._opacity = opacity
        self.state.opacity = opacity

        self._selection_actor = None
        self._filter_actor = None

        self._selection_color = selection_color
        self._rel_selection_opacity = rel_selection_opacity
        self._rel_filter_opacity = rel_filter_opacity

    @property
    def visibility(self):
        return self._visibility

    @visibility.setter
    def visibility(self, value):
        self.actor.visibility = value
        if self._selection_actor is not None:
            self._selection_actor.visibility = value

        if self._filter_actor is not None:
            self._filter_actor.visibility = value

        self.plotter.render()

        self._visibility = value
        self.state.visibility = value

    @property
    def opacity(self):
        return self._opacity

    @opacity.setter
    def opacity(self, value):
        if value < 0 or value > 1:
            raise ValueError("opacity must be between 0 and 1")

        if self._filter_actor is not None:
            self.actor.prop.opacity = value * self._rel_filter_opacity
            self._filter_actor.prop.opacity = value
        else:
            self.actor.prop.opacity = value

        if self._selection_actor is not None:
            self._selection_actor.prop.opacity = value * self._rel_selection_opacity

        self.plotter.render()

        self._opacity = value
        self.state.opacity = value

    @property
    def selection_actor(self):
        return self._selection_actor

    @property
    def selection_color(self):
        return self._selection_color

    @selection_color.setter
    def selection_color(self, value):
        if self._selection_actor is not None:
            self._selection_actor.prop.color = value
            self.plotter.render()

        self._selection_color = value

    @property
    def rel_selection_opacity(self):
        return self._rel_selection_opacity

    @rel_selection_opacity.setter
    def rel_selection_opacity(self, value):
        if self._selection_actor is not None:
            self._selection_actor.prop.opacity = self._opacity * value
            self.plotter.render()

        self._rel_selection_opacity = value

    @property
    def filter_actor(self):
        return self._filter_actor

    @property
    def rel_filter_opacity(self):
        return self._rel_filter_opacity

    @rel_filter_opacity.setter
    def rel_filter_opacity(self, value):
        if self._filter_actor is not None:
            self.actor.prop.opacity = self._opacity * value
            self.plotter.render()

        self._rel_filter_opacity = value


class _PointLayer(_BaseLayer):
    def __init__(
        self,
        name,
        mesh,
        actor,
        plotter,
        pickable=True,
        accelerated_picking=False,
        point_size=15,
        rel_selected_point_size=1.1,
        rel_filtered_point_size=1.1,
        *args,
        **kwargs,
    ):
        super().__init__(name, mesh, actor, plotter, *args, **kwargs)

        self.pickable = pickable
        self.accelerated_picking = accelerated_picking

        self.picker = None
        self.filter_picker = None

        self._picked_point = None
        self._selected_points = []
        self._filtered_points = []
        self._boolean_filter = None

        if self.pickable:
            self._make_pickable()

        self.point_size = point_size
        self.rel_selected_point_size = rel_selected_point_size
        self.rel_filtered_point_size = rel_filtered_point_size

        self.n_points = mesh.n_points

    def _make_pickable(self):
        self.picker = vtkPointPicker()
        self.picker.SetTolerance(0.005)

        self.filter_picker = vtkPointPicker()
        self.filter_picker.SetTolerance(0.005)

        if self.accelerated_picking == True:
            for picker in [self.picker, self.filter_picker]:
                # add locator for acceleration
                locator = vtkPointLocator()
                locator.SetDataSet(self.mesh)
                locator.BuildLocator()
                picker.AddLocator(locator)

                # use hardware selection for acceleration
                hw_selector = vtkHardwareSelector()
                hw_selector.SetFieldAssociation(vtkDataObject.FIELD_ASSOCIATION_POINTS)
                hw_selector.SetRenderer(self.plotter.renderer)

    def _make_selection_by_pick(self, pos, actor):
        if actor == self.actor:
            picker = self.picker
            on_filter = False

        elif actor == self.filter_actor:
            picker = self.filter_picker
            on_filter = True

        picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)

        picked_point = picker.GetPointId()
        if picked_point is not None:
            if picked_point == -1:
                return
            else:
                shift_pressed = self.plotter.iren.interactor.GetShiftKey()
                ctrl_pressed = self.plotter.iren.interactor.GetControlKey()

                if shift_pressed:
                    self._make_continuous_multi_selection(
                        picked_point, on_filter=on_filter
                    )
                elif ctrl_pressed:
                    self._make_discontinuous_multi_selection(
                        picked_point, on_filter=on_filter
                    )

                else:
                    self._make_single_selection(picked_point, on_filter=on_filter)

                if on_filter:
                    picked_point = self.filtered_points[picked_point]

                self._picked_point = picked_point

    def _make_single_selection(self, picked_point, on_filter=False):
        if on_filter:
            picked_point = self.filtered_points[picked_point]

        self.selected_points = [picked_point]

    def _make_discontinuous_multi_selection(self, picked_point, on_filter=False):
        pass

    def _make_continuous_multi_selection(self, picked_point, on_filter=False):
        pass  # not trivial as cell IDs are not inherently sequential along hole

    def _update_selection_object(self):
        if len(self.selected_points) != 0:
            selection_mesh = self.mesh.extract_points(self.selected_points)
            if (selection_mesh.n_points != 0) and (selection_mesh.n_cells != 0):
                selection_actor = self.actor.copy(deep=True)
                selection_actor.mapper.dataset = selection_mesh

                self.plotter.add_actor(
                    selection_actor,
                    name=self.name + " selection",
                    pickable=False,
                    reset_camera=False,
                )

                self._selection_actor = selection_actor

                # update selection actor properties
                self.selection_color = self.selection_color
                self.opacity = self.opacity
                selection_actor.prop.point_size = (
                    self.point_size * self.rel_selected_point_size
                )

    def _reset_selection(self):
        self._picked_point = None
        self.selected_points = []

        self.plotter.remove_actor(self._selection_actor)
        self._selection_actor = None

    @property
    def picked_point(self):
        return self._picked_point

    @property
    def selected_points(self):
        return self._selected_points

    @selected_points.setter
    def selected_points(self, points):
        try:
            points = convert_to_numpy_array(points)
        except:
            raise ValueError(
                "Points must be a sequence, pandas object, or numpy array."
            )

        if (points > self.n_points).any() or (points < 0).any():
            raise ValueError(
                f"Points must be between 0 and the number of points in the dataset, {self.n_points - 1}."
            )

        self._selected_points = points

        self._update_selection_object()

    @property
    def selected_ids(self):
        return self.selected_points

    @selected_ids.setter
    def selected_ids(self, ids):
        self.selected_points = ids

    @property
    def boolean_filter(self):
        return self._boolean_filter

    @boolean_filter.setter
    def boolean_filter(self, value):
        try:
            boolean_filter = convert_to_numpy_array(value)
        except:
            raise ValueError(
                "Boolean filter must be a sequence, pandas object, or numpy array."
            )

        if len(boolean_filter) != self.n_points:
            raise ValueError(
                f"Boolean filter must have the same length as the number of points in the dataset, {self.n_points}."
            )

        self._boolean_filter = boolean_filter
        self._filtered_points = np.arange(self.n_points)[boolean_filter]

        self._update_filter_object()

    @property
    def filtered_points(self):
        return self._filtered_points

    @property
    def filtered_ids(self):
        return self._filtered_points

    def _update_filter_object(self):
        filter_mesh = self.mesh.extract_points(self.filtered_points)
        if (filter_mesh.n_points != 0) and (filter_mesh.n_cells != 0):
            filter_actor = self.actor.copy(deep=True)
            filter_actor.mapper.dataset = filter_mesh
            self.plotter.add_actor(
                filter_actor,
                name=self.name + " filter",
                pickable=True,
                reset_camera=False,
            )

            self._filter_actor = filter_actor

            # make filtered out intervals not pickable
            self.actor.SetPickable(False)

            # update opacity and visibility
            self.opacity = self.opacity
            self.visibility = self.visibility

    def _reset_filter(self):
        self._filtered_points = []
        self.plotter.remove_actor(self._filter_actor)
        self._filter_actor = None

        # make previously filtered out intervals pickable
        self.actor.SetPickable(True)

        # return previously filtered out intervals to original opacity
        self.opacity = self.opacity

    def convert_selection_to_filter(self):
        boolean_filter = np.isin(np.arange(self.n_points), self.selected_points)
        self._reset_selection()
        self.boolean_filter = boolean_filter

    def convert_filter_to_selection(self, keep_filter=False):
        self.selected_points = self.filtered_points

        if keep_filter == False:
            self._reset_filter()

    @property
    def all_ids(self):
        return np.arange(self.n_points)


class _IntervalLayer(_BaseLayer):
    def __init__(
        self,
        name,
        mesh,
        actor,
        plotter,
        pickable=True,
        accelerated_picking=False,
        n_sides=20,
        *args,
        **kwargs,
    ):
        super().__init__(name, mesh, actor, plotter, *args, **kwargs)

        self.pickable = pickable
        self.accelerated_picking = accelerated_picking

        self.picker = None
        self.filter_picker = None

        self._picked_cell = None
        self._selected_cells = []
        self._selected_intervals = []
        self._filtered_cells = []
        self._filtered_intervals = []
        self._boolean_filter = None

        if self.pickable:
            self._make_pickable()

        self.n_sides = n_sides
        self.n_intervals = int(mesh.n_cells / n_sides)

    def _make_pickable(self):
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(0.0005)

        self.filter_picker = vtkCellPicker()
        self.filter_picker.SetTolerance(0.0005)

        if self.accelerated_picking == True:
            for picker in [self.picker, self.filter_picker]:
                # add locator for acceleration
                locator = vtkCellLocator()
                locator.SetDataSet(self.mesh)
                locator.BuildLocator()
                picker.AddLocator(locator)

                # use hardware selection for acceleration
                hw_selector = vtkHardwareSelector()
                hw_selector.SetFieldAssociation(vtkDataObject.FIELD_ASSOCIATION_CELLS)
                hw_selector.SetRenderer(self.plotter.renderer)

    def _make_selection_by_pick(self, pos, actor):
        if actor == self.actor:
            picker = self.picker
            on_filter = False

        elif actor == self.filter_actor:
            picker = self.filter_picker
            on_filter = True

        picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)

        picked_cell = picker.GetCellId()
        if picked_cell is not None:
            if picked_cell == -1:
                return
            else:
                shift_pressed = self.plotter.iren.interactor.GetShiftKey()
                ctrl_pressed = self.plotter.iren.interactor.GetControlKey()

                if shift_pressed:
                    self._make_continuous_multi_selection(
                        picked_cell, on_filter=on_filter
                    )
                elif ctrl_pressed:
                    self._make_discontinuous_multi_selection(
                        picked_cell, on_filter=on_filter
                    )

                else:
                    self._make_single_selection(picked_cell, on_filter=on_filter)

                if on_filter:
                    picked_cell = self._filtered_cells[picked_cell]

                self._picked_cell = picked_cell

    def _make_single_selection(self, picked_cell, on_filter=False):
        cells_per_interval = self.n_sides

        selected_interval = int(np.floor(picked_cell / cells_per_interval))

        selected_cells = np.arange(
            selected_interval * cells_per_interval,
            (selected_interval + 1) * cells_per_interval,
        ).tolist()

        if on_filter:
            selected_interval = self.filtered_intervals[selected_interval]
            selected_cells = list(self._filtered_cells[selected_cells])

        self.selected_intervals = [selected_interval]
        self._selected_cells = selected_cells

    def _make_discontinuous_multi_selection(self, picked_cell, on_filter=False):
        cells_per_interval = self.n_sides

        selected_interval = int(np.floor(picked_cell / cells_per_interval))
        selected_cells = np.arange(
            selected_interval * cells_per_interval,
            (selected_interval + 1) * cells_per_interval,
        ).tolist()

        if on_filter:
            selected_interval = self.filtered_intervals[selected_interval]
            selected_cells = list(self._filtered_cells[selected_cells])

        self.selected_intervals += [selected_interval]
        self._selected_cells += selected_cells

    def _make_continuous_multi_selection(self, picked_cell, on_filter=False):
        cells_per_interval = self.n_sides

        if on_filter:
            prev_picked_cell = np.where(self._filtered_cells == self._picked_cell)[0][0]
            prev_selected_intervals = np.where(
                np.isin(self.filtered_intervals, self.selected_intervals[:-1])
            )[0].tolist()
            prev_selected_intervals += np.where(
                np.isin(self.filtered_intervals, self.selected_intervals[-1])
            )[
                0
            ].tolist()  #  needed as np.isin or np.where seems to sort the output and the resulting first interval should be last

        else:
            prev_picked_cell = self._picked_cell
            prev_selected_intervals = self.selected_intervals

        if prev_picked_cell is not None:
            if prev_picked_cell < picked_cell:  # normal direction (down the hole)
                selected_intervals = np.arange(
                    prev_selected_intervals[-1] + 1,
                    int(np.floor(picked_cell / cells_per_interval)) + 1,
                ).tolist()
                selected_cells = np.arange(
                    (selected_intervals[0]) * cells_per_interval,
                    (selected_intervals[-1] + 1) * cells_per_interval,
                ).tolist()

                if on_filter:
                    selected_intervals = list(
                        self.filtered_intervals[selected_intervals]
                    )
                    selected_cells = list(self._filtered_cells[selected_cells])

                self.selected_intervals += selected_intervals
                self._selected_cells += selected_cells

            else:  # reverse direction (up the hole)
                selected_intervals = np.arange(
                    int(np.floor(picked_cell / cells_per_interval)),
                    prev_selected_intervals[-1],
                ).tolist()
                selected_cells = np.arange(
                    (selected_intervals[0] * cells_per_interval),
                    (selected_intervals[-1] + 1) * cells_per_interval,
                ).tolist()

                if on_filter:
                    selected_intervals = list(
                        self.filtered_intervals[selected_intervals]
                    )
                    selected_cells = list(self._filtered_cells[selected_cells])

                self.selected_intervals = selected_intervals + self.selected_intervals
                self._selected_cells = selected_cells + self._selected_cells

    @property
    def picked_cell(self):
        return self._picked_cell

    @property
    def selected_cells(self):
        return self._selected_cells

    @property
    def selected_intervals(self):
        return self._selected_intervals

    @selected_intervals.setter
    def selected_intervals(self, intervals):
        try:
            intervals = convert_to_numpy_array(intervals)
        except:
            raise ValueError(
                "Intervals must be a sequence, pandas object, or numpy array."
            )

        if (intervals > self.n_intervals).any() or (intervals < 0).any():
            raise ValueError(
                f"Intervals must be between 0 and the number of intervals in the dataset, {self.n_intervals - 1}."
            )

        intervals = intervals.tolist()
        interval_cells = []
        cells_per_interval = self.n_sides
        for interval in intervals:
            interval_cells += np.arange(
                interval * cells_per_interval,
                (interval + 1) * cells_per_interval,
            ).tolist()

        self._selected_intervals = intervals
        self._selected_cells = interval_cells

        self._update_selection_object()

    @property
    def selected_ids(self):
        return self.selected_intervals

    @selected_ids.setter
    def selected_ids(self, ids):
        self.selected_intervals = ids

    def _update_selection_object(self):
        if len(self._selected_cells) != 0:
            selection_mesh = self.mesh.extract_cells(self._selected_cells)
            if (selection_mesh.n_points != 0) and (selection_mesh.n_cells != 0):
                selection_actor = self.plotter.add_mesh(
                    selection_mesh,
                    name=self.name + " selection",
                    color=self.selection_color,
                    opacity=self.opacity * self.rel_selection_opacity,
                    reset_camera=False,
                    pickable=False,
                )
                self._selection_actor = selection_actor

                if selection_actor is not None:
                    selection_actor.mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
                        0, -6
                    )
                    self.plotter.render()

    def _reset_selection(self):
        self._picked_cell = None
        self._selected_cells = []
        self.selected_intervals = []

        self.plotter.remove_actor(self._selection_actor)
        self._selection_actor = None

    @property
    def boolean_filter(self):
        return self._boolean_filter

    @boolean_filter.setter
    def boolean_filter(self, value):
        try:
            boolean_filter = convert_to_numpy_array(value)
        except:
            raise ValueError(
                "Boolean filter must be a sequence, pandas object, or numpy array."
            )

        if len(boolean_filter) != self.n_intervals:
            raise ValueError(
                f"Boolean filter must have the same length as the number of intervals in the dataset, {self.n_intervals}."
            )

        self._boolean_filter = boolean_filter
        self._filtered_intervals = np.arange(self.n_intervals)[boolean_filter]

        cells_per_interval = self.n_sides
        boolean_cell_filter = np.repeat(boolean_filter, cells_per_interval)
        self._filtered_cells = np.arange(self.n_intervals * cells_per_interval)[
            boolean_cell_filter
        ]

        self._update_filter_object()

    @property
    def filtered_cells(self):
        return self._filtered_cells

    @property
    def filtered_intervals(self):
        return self._filtered_intervals

    @property
    def filtered_ids(self):
        return self._filtered_intervals

    def _update_filter_object(self):
        filter_mesh = self.mesh.extract_cells(self._filtered_cells)
        if (filter_mesh.n_points != 0) and (filter_mesh.n_cells != 0):
            filter_actor = self.actor.copy(deep=True)
            filter_actor.mapper.dataset = filter_mesh
            self.plotter.add_actor(
                filter_actor,
                name=self.name + " filter",
                pickable=True,
                reset_camera=False,
            )

            self._filter_actor = filter_actor

            # make filtered out intervals not pickable
            self.actor.SetPickable(False)

            # update opacity and visibility
            self.opacity = self.opacity
            self.visibility = self.visibility

    def _reset_filter(self):
        self._filtered_cells = []
        self._filtered_intervals = []
        self.plotter.remove_actor(self._filter_actor)
        self._filter_actor = None

        # make previously filtered out intervals pickable
        self.actor.SetPickable(True)

        # return previously filtered out intervals to original opacity
        self.opacity = self.opacity

    def convert_selection_to_filter(self):
        boolean_filter = np.isin(np.arange(self.n_intervals), self.selected_intervals)
        self._reset_selection()
        self.boolean_filter = boolean_filter

    def convert_filter_to_selection(self, keep_filter=False):
        self._selected_cells = self._filtered_cells
        self.selected_intervals = self.filtered_intervals

        if keep_filter == False:
            self._reset_filter()

    @property
    def all_ids(self):
        return np.arange(self.n_intervals)


class _DataLayer(ImageMixin, _BaseLayer, Plotting2dMixin):
    def __init__(
        self,
        name,
        mesh,
        actor,
        plotter,
        *args,
        **kwargs,
    ):
        super().__init__(name, mesh, actor, plotter, *args, **kwargs)

        self._active_array_name = mesh.active_scalars_name
        self.state.active_array_name = self._active_array_name

        if hasattr(
            actor.mapper.lookup_table.cmap, "name"
        ):  # only set if active_array_name is continuous
            self._cmap = actor.mapper.lookup_table.cmap.name
            self.state.cmap = self._cmap
            self._clim = actor.mapper.lookup_table.scalar_range
            self.state.clim = self._clim
        else:
            self._cmap = None
            self.state.cmap = self._cmap
            self._clim = None
            self.state.clim = self._clim

        self._cmaps = plt.colormaps()

        self._continuous_array_names = []
        self._categorical_array_names = []

        # decoding categorical data
        self.code_to_hole_id_map = {}
        self.code_to_cat_map = {}

        # categorical data cmaps
        self.cat_to_color_map = {}
        self.matplotlib_formatted_color_maps = None

        # to aid w resetting when switching btwn arrays of diff. types
        self.preceding_array_type = None

        # active image viewer
        self.im_viewer = None

    @property
    def active_array_name(self):
        return self._active_array_name

    @active_array_name.setter
    def active_array_name(self, value):
        if value not in self.array_names:
            raise ValueError(f"{value} is not an array name.")

        self._active_array_name = value
        self.state.active_array_name = value

        self.actor.mapper.dataset.set_active_scalars(value)
        if self._filter_actor is not None:
            self.filter_actor.mapper.dataset.set_active_scalars(value)

        if (value in self.continuous_array_names) and (
            self.preceding_array_type != "continuous"
        ):
            self.cmap = self.cmap
            self.clim = self.clim

        elif value in self.categorical_array_names:
            if value not in self.matplotlib_formatted_color_maps:
                raise ValueError(f"{value} is not a valid colormap.")

            cmap = self.matplotlib_formatted_color_maps[value]
            self.actor.mapper.lookup_table = pv.LookupTable(cmap)
            self.actor.mapper.lookup_table.scalar_range = [
                0,
                len(self.cat_to_color_map[value]) - 1,
            ]
            self.actor.mapper.SetUseLookupTableScalarRange(True)

            if self._filter_actor is not None:
                self.filter_actor.mapper.lookup_table = pv.LookupTable(cmap)
                self.filter_actor.mapper.lookup_table.scalar_range = [
                    0,
                    len(self.cat_to_color_map[value]) - 1,
                ]
                self.filter_actor.mapper.SetUseLookupTableScalarRange(True)

        self.plotter.render()

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, value):
        if self.active_array_name in self.continuous_array_names:
            self.actor.mapper.lookup_table.cmap = value
            if self._filter_actor is not None:
                self.filter_actor.mapper.lookup_table.cmap = value

            self.plotter.render()

            self._cmap = value
            self.state.cmap = value

    @property
    def clim(self):
        return self._clim

    @clim.setter
    def clim(self, value):
        if self.active_array_name in self.continuous_array_names:
            self.actor.mapper.lookup_table.scalar_range = value
            self.actor.mapper.SetUseLookupTableScalarRange(True)
            if self._filter_actor is not None:
                self.filter_actor.mapper.lookup_table.scalar_range = value
                self.filter_actor.mapper.SetUseLookupTableScalarRange(True)

            self.plotter.render()

            self._clim = value
            self.state.clim = value

    @property
    def array_names(self):
        return self.mesh.array_names

    @property
    def continuous_array_names(self):
        return self._continuous_array_names

    @property
    def categorical_array_names(self):
        return self._categorical_array_names

    def data_within_interval(self, hole_id, interval):
        pass

    @property
    def selected_data(self):
        ids = self.selected_ids
        data = self._process_data_output(ids)

        return data

    @property
    def selected_hole_ids(self):
        data = self.all_data
        hole_ids = data["hole ID"][self.selected_ids]
        hole_ids = list(hole_ids.unique().tolist())  # dbl list necessary

        return hole_ids

    @selected_hole_ids.setter
    def selected_hole_ids(self, hole_ids):
        if isinstance(hole_ids, str):
            hole_ids = [hole_ids]

        if not isinstance(hole_ids, (list, np.ndarray, pd.Series)):
            raise ValueError("hole_ids must be a list, numpy array, or pandas Series.")

        data = self.all_data
        if self.filter_actor is not None:
            data = data[self.boolean_filter]

        matches = data["hole ID"].isin(hole_ids)
        self.selected_ids = list(data.loc[matches].index.tolist())  # dbl list necessary
        self._selected_hole_ids = hole_ids

    @property
    def filtered_data(self):
        ids = self.filtered_ids
        data = self._process_data_output(ids)

        return data

    @property
    def filtered_hole_ids(self):
        data = self.all_data
        hole_ids = data["hole ID"][self.filtered_ids]
        hole_ids = hole_ids.unique().tolist()

        return hole_ids

    @property
    def all_data(self):
        ids = self.all_ids
        data = self._process_data_output(ids)

        return data

    def _process_data_output(self, ids, array_names=[], step=1):
        if len(array_names) == 0:
            array_names = self.mesh.array_names

        data_dict = {}
        for name in array_names:
            data_dict[name] = self.mesh[name][::step][ids]

        data = pd.DataFrame(data_dict)

        # decode hole IDs
        data["hole ID"] = [self.code_to_hole_id_map[code] for code in data["hole ID"]]

        # decode categorical data
        for name in self.categorical_array_names:
            data[name] = data[name].astype("category")
            data[name] = [self.code_to_cat_map[name][code] for code in data[name]]

        return data


class PointDataLayer(_DataLayer, _PointLayer):
    def __init__(self, name, mesh, actor, plotter, *args, **kwargs):
        super().__init__(name, mesh, actor, plotter, *args, **kwargs)

    def __getitem__(self, key):
        return self.all_data[key]

    def __setitem__(self, key, value):
        self.mesh[key] = value
        if self.filter_actor is not None:
            self.filter_actor.mapper.dataset[key] = value[self.boolean_filter]
        else:
            self.mesh[key] = value

        self.active_array_name = key
        self.array_names.append(key)
        if np.issubdtype(value.dtype, np.number):
            self.continuous_array_names.append(key)
        else:
            self.categorical_array_names.append(key)

    def _process_data_output(self, ids, array_names=[]):
        exclude = ["vtkOriginalPointIds", "vtkOriginalCellIds"]  # added by pyvista
        array_names = [name for name in self.mesh.array_names if name not in exclude]

        data = super()._process_data_output(ids, array_names)

        return data

    def selected_drill_log(
        self,
        log_array_names=[],
    ):
        data = self.selected_data

        hole_id = self.selected_hole_ids
        if len(hole_id) != 1:
            raise ValueError(
                "Drill log can only be created for a single hole at a time."
            )

        # check if no variables are passed; if so, use all variables
        if len(log_array_names) == 0:
            log_array_names = self.categorical_array_names + self.continuous_array_names

        log = DrillLog()

        # add point data
        depths = data[["from", "to"]].values

        for array_name in log_array_names:
            if array_name in self.categorical_array_names:
                cat_to_color_map = self.cat_to_color_map
                values = data[array_name].values
                log.add_categorical_point_data(
                    array_name,
                    depths,
                    values,
                    cat_to_color_map.get(array_name, None),
                )

            elif array_name in self.continuous_array_names:
                values = data[array_name].values
                log.add_continuous_point_data(array_name, depths, values)

            else:
                raise ValueError(f"Data for array with name {array_name} not present.")

        log.create_figure(y_axis_label="Depth (m)", title=hole_id[0])

        return log.fig


class IntervalDataLayer(_DataLayer, _IntervalLayer):
    def __init__(self, name, mesh, actor, plotter, *args, **kwargs):
        super().__init__(name, mesh, actor, plotter, *args, **kwargs)

    def __getitem__(self, key):
        return self.all_data[key]

    def __setitem__(self, key, value):
        cells_per_interval = self.n_sides
        value = np.repeat(value, cells_per_interval)
        self.mesh[key] = value
        if self.filter_actor is not None:
            boolean_filter = np.repeat(self.boolean_filter, cells_per_interval)
            self.filter_actor.mapper.dataset[key] = value[boolean_filter]

        self.active_array_name = key
        self.array_names.append(key)
        if np.issubdtype(value.dtype, np.number):
            self.continuous_array_names.append(key)
        else:
            self.categorical_array_names.append(key)

    def _make_selection_by_dbl_click_pick(self, pos, actor):
        if actor == self.actor:
            picker = self.picker
            on_filter = False

        elif actor == self.filter_actor:
            picker = self.filter_picker
            on_filter = True

        picker.Pick(pos[0], pos[1], 0, self.plotter.renderer)

        picked_cell = picker.GetCellId()
        if picked_cell is not None:
            if picked_cell == -1:
                return
            else:
                ctrl_pressed = self.plotter.iren.interactor.GetControlKey()

                if ctrl_pressed:
                    self._make_multi_selection_by_dbl_click_pick(
                        picked_cell, on_filter=on_filter
                    )
                else:
                    self._make_single_selection_by_dbl_click_pick(
                        picked_cell, on_filter=on_filter
                    )
                    # self._make_single_selection(picked_cell, on_filter=on_filter)

                if on_filter:
                    picked_cell = self._filtered_cells[picked_cell]

                self._picked_cell = picked_cell

    def _make_single_selection_by_dbl_click_pick(self, picked_cell, on_filter=False):
        self._make_single_selection(picked_cell, on_filter=on_filter)
        self.selected_hole_ids = self.selected_hole_ids

    def _make_multi_selection_by_dbl_click_pick(self, picked_cell, on_filter=False):
        prev_selected_intervals = self.selected_intervals
        prev_selected_cells = self.selected_cells

        self._make_single_selection_by_dbl_click_pick(picked_cell, on_filter=on_filter)

        self.selected_intervals += prev_selected_intervals
        self._selected_cells += prev_selected_cells

    def _process_data_output(self, ids, array_names=[]):
        exclude = [
            "TubeNormals",
            "vtkOriginalPointIds",
            "vtkOriginalCellIds",
        ]  # added by pvista; first has excess dims
        array_names = [name for name in self.mesh.array_names if name not in exclude]

        data = super()._process_data_output(ids, array_names, step=self.n_sides)

        return data

    def selected_drill_log(
        self,
        log_array_names=[],
    ):
        data = self.selected_data

        hole_id = self.selected_hole_ids
        if len(hole_id) != 1:
            raise ValueError(
                "Drill log can only be created for a single hole at a time."
            )

        # check if no variables are passed; if so, use all variables
        if len(log_array_names) == 0:
            log_array_names = self.categorical_array_names + self.continuous_array_names

        log = DrillLog()

        # add interval data
        depths = data[["from", "to"]].values

        for array_name in log_array_names:
            if array_name in self.categorical_array_names:
                cat_to_color_map = self.cat_to_color_map
                values = data[array_name].values
                log.add_categorical_interval_data(
                    array_name,
                    depths,
                    values,
                    cat_to_color_map.get(array_name, None),
                )

            elif array_name in self.continuous_array_names:
                values = data[array_name].values
                log.add_continuous_interval_data(array_name, depths, values)

            else:
                raise ValueError(f"Data for array with name {array_name} not present.")

        log.create_figure(y_axis_label="Depth (m)", title=hole_id[0])

        return log.fig

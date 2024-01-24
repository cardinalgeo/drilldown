from vtk import (
    vtkCellPicker,
    vtkPointPicker,
    vtkCellLocator,
    vtkPointLocator,
    vtkHardwareSelector,
    vtkDataObject,
)
import numpy as np

from ..utils import convert_to_numpy_array


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
        self.name = name
        self.mesh = mesh
        self.actor = actor
        self._visibility = visibility
        self._opacity = opacity
        # self._opacity_while_not_visible = opacity
        self.plotter = plotter

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
        visibility=True,
        opacity=1,
        pickable=True,
        accelerated_picking=False,
        point_size=15,
        rel_selected_point_size=1.1,
        rel_filtered_point_size=1.1,
    ):
        super().__init__(name, mesh, actor, plotter, visibility, opacity)

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

        self._selected_points = [picked_point]

    def _make_discontinuous_multi_selection(self, picked_point, on_filter=False):
        pass

    def _make_continuous_multi_selection(self, picked_point, on_filter=False):
        pass  # not trivial as cell IDs are not inherently sequential along hole

    def _update_selection_object(self):
        selection_mesh = self.mesh.extract_points(self.selected_points)
        if (selection_mesh.n_points != 0) and (selection_mesh.n_cells != 0):
            selection_actor = self.plotter.add_mesh(
                selection_mesh,
                self.name + " selection",
                color=self.selection_color,
                opacity=self.opacity * self.rel_selection_opacity,
                point_size=self.point_size * self.rel_selected_point_size,
                render_points_as_spheres=True,
                reset_camera=False,
                pickable=False,
            )

            self._selection_actor = selection_actor
            if selection_actor is not None:
                selection_actor.mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(
                    0, -6
                )
                self.plotter.render()

            return selection_actor

    def _reset_selection(self):
        self._picked_point = None
        self._selected_points = []

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

    def _update_filter_object(self):
        filter_mesh = self.mesh.extract_points(self.filtered_points)
        if (filter_mesh.n_points != 0) and (filter_mesh.n_cells != 0):
            filter_actor = self.actor.copy(deep=True)
            filter_actor.mapper.dataset = filter_mesh
            self.plotter.add_actor(filter_actor, reset_camera=False)

            self._filter_actor = filter_actor

            # make filtered out intervals not pickable
            self.actor.SetPickable(False)

            # update opacity and visibility
            self.opacity = self.opacity
            self.visibility = self.visibility

            # self.plotter.render()

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


class _IntervalLayer(_BaseLayer):
    def __init__(
        self,
        name,
        mesh,
        actor,
        plotter,
        visibility=True,
        opacity=1,
        pickable=True,
        accelerated_picking=False,
        n_sides=20,
    ):
        super().__init__(name, mesh, actor, plotter, visibility, opacity)

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

        self._selected_intervals = [selected_interval]
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

        self._selected_intervals += [selected_interval]
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

                self._selected_intervals += selected_intervals
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

                self._selected_intervals = selected_intervals + self.selected_intervals
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

    def _update_selection_object(self):
        selection_mesh = self.mesh.extract_cells(self._selected_cells)
        if (selection_mesh.n_points != 0) and (selection_mesh.n_cells != 0):
            selection_actor = self.plotter.add_mesh(
                selection_mesh,
                self.name + " selection",
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

            return selection_actor

    def _reset_selection(self):
        self._picked_cell = None
        self._selected_cells = []
        self._selected_intervals = []

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

            # self.plotter.render()

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


class _DataLayer(_BaseLayer):
    def __init__(
        self,
        name,
        mesh,
        actor,
        plotter,
        visibility=True,
        opacity=1,
        pickable=True,
        active_var=None,
        cmap=None,
        clim=None,
    ):
        super().__init__(name, mesh, actor, plotter, visibility, opacity, pickable)

        self._active_var = active_var
        self._cmap = cmap
        self._clim = clim

        self._continuous_array_names = []
        self._categorical_array_names = []

    def __getitem__(self, key):
        return self.mesh[key]

    def __setitem__(self, key, value):
        self.mesh[key] = value
        self.mesh.keys()

    def array_names(self):
        return self.mesh.array_names

    @property
    def continuous_array_names(self):
        return self._continuous_array_names

    @continuous_array_names.setter
    def continuous_array_names(self, value):
        self._continuous_array_names = value

    @property
    def categorical_array_names(self):
        return self._categorical_array_names

    @categorical_array_names.setter
    def categorical_array_names(self, value):
        self._categorical_array_names = value

    def data_within_interval(self, hole_id, interval):
        pass


class PointDataLayer(_DataLayer, _PointLayer):
    def __init__(
        self,
        name,
        mesh,
        actor,
        plotter,
        visibility=True,
        opacity=1,
        pickable=True,
        active_var=None,
        cmap=None,
        clim=None,
    ):
        super().__init__(
            name,
            mesh,
            actor,
            plotter,
            visibility,
            opacity,
            pickable,
            active_var,
            cmap,
            clim,
        )


class IntervalDataLayer(_DataLayer, _IntervalLayer):
    def __init__(
        self,
        name,
        mesh,
        actor,
        plotter,
        visibility=True,
        opacity=1,
        pickable=True,
        active_var=None,
        cmap=None,
        clim=None,
    ):
        super().__init__(
            name,
            mesh,
            actor,
            plotter,
            visibility,
            opacity,
            pickable,
            active_var,
            cmap,
            clim,
        )

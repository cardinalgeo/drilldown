from vtk import (
    vtkPicker,
    vtkPropPicker,
    vtkMapper,
)
import pyvista as pv
from pyvista.trame.jupyter import show_trame
from trame.app import get_server

from IPython.display import IFrame
import uuid

from .layer.layer import IntervalDataLayer, PointDataLayer
from .layer.layer_list import LayerList
from .utils import is_jupyter


def is_numeric_tuple(tup):
    """Check if all elements of a tuple are numeric."""
    return all(isinstance(x, (int, float)) for x in tup)


def actors_collection_to_list(actors_collection):
    """Convert a vtk.vtkActorCollection to a list of vtk.vtkActor objects."""
    actors_collection.InitTraversal()
    actors_list = []
    for i in range(actors_collection.GetNumberOfItems()):
        actors_list.append(actors_collection.GetNextActor())

    return actors_list


class Plotter(pv.Plotter):
    """Plotting object for displaying drillholes and related datasets. To be used by the :class:`drilldown.DrillDownPlotter` (with GUI) or on its own (without GUI).

    Parameters
    ----------
    *args : tuple
        Positional arguments to pass to the superclass.

    **kwargs : dict
        Keyword arguments to pass to the superclass.
    """

    def __init__(self, *args, **kwargs):
        """Initialize plotter."""

        super().__init__(*args, **kwargs)

        self.height = 600
        self.set_background("white")
        self.enable_trackball_style()
        vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()

        self.layers = LayerList()

        # track datasets
        self.collars = None
        self.surveys = None
        self.intervals = []
        self.points = []

        # track collar and survey actors, in absence of layer classes
        self.collar_actor = None
        self.collar_label_actor = None
        self.survey_actor = None

        # pickers for pIcKiNg
        self.actor_picker = vtkPropPicker()
        self.actors_to_make_not_pickable_picker = vtkPicker()

        # track layer corresponding to picked actor
        self.picked_layer = None

        # track clicks
        self.track_click_position(side="left", callback=self._on_single_click)
        self.track_click_position(
            side="left", callback=self._on_double_click, double=True
        )

        # track active selections and filters across layers
        self._active_selections_and_filters = []
        # spin up a server to manager GUI, if present
        self.server = get_server(str(uuid.uuid4()))
        self.state = self.server.state
        self.server.client_type = "vue2"
        self.state.layer_names = []

        # track visibility and opacity of collars, collar labels, and surveys
        self._collar_visibility = True
        self._collar_opacity = 1
        self._collar_label_visibility = False
        self._survey_visibility = True
        self._survey_opacity = 1

    def add_collars(
        self,
        collars,
        show_labels=True,
        opacity=1,
        point_size=15,
        visibility=True,
        *args,
        **kwargs,
    ):
        """Add a `drilldown.Collars` object (i.e., containing the x-y-z coordinates of the tops of each hole) to the plotter.

        If not already constructed, the mesh for the collars will be created by the `drilldown.Collars` object and added.

        Parameters
        ----------
        collars : drilldown.Collars
            The collars to add to the plotter.

        show_labels : bool, optional
            Whether to show labels (i.e., hole IDs) for the collars. Defaults to True.

        opacity : float, optional
            The opacity of the collars. Defaults to 1.

        point_size : int or float, optional
            The size of the points representing the collars. Defaults to 15.

        *args : tuple
            Positional arguments to pass to `add_mesh()`.

        **kwargs : dict
            Keyword arguments to pass to `add_mesh()`.

        Returns
        -------
        pyvista.Actor or tuple
            The actor representing the collars, and optionally the actor representing the labels.

        Examples
        --------
        Create a `drilldown.Collars` object and add it to the plotter:

        >>> import drilldown as dd
        >>> data_dict = dd.examples.load_tom_zone_macpass_project()
        >>> collar_data = data_dict["collar"]
        >>> collars = dd.Collars()
        >>> collars.add_data(collar_data["hole_ID"], collar_data[["x", "y", "z"]])
        >>> pl = dd.Plotter()
        >>> pl.add_collars(collars)
        >>> pl.show()

        """

        from .holes import Collars

        if not isinstance(collars, Collars):
            raise TypeError("collars must be a DrillDown.Collars object.")

        self.collars = collars
        name = "collars"
        if collars.mesh is None:
            collars.make_mesh()

        mesh = collars.mesh
        actor = self.add_mesh(
            mesh,
            name=name,
            opacity=opacity,
            render_points_as_spheres=True,
            point_size=point_size,
            pickable=False,
            *args,
            **kwargs,
        )
        self.collar_actor = actor
        self._collar_visibility = visibility
        self._collar_opacity = opacity

        if show_labels == True:
            label_actor = self.add_point_labels(
                mesh, mesh["hole ID"], shape_opacity=0.5, show_points=False
            )
            self.collar_label_actor = label_actor
            self._collar_label_visibility = visibility

            return actor, label_actor

        else:
            return actor

    def add_surveys(self, surveys, opacity=1, *args, **kwargs):
        """Add a `drilldown.Surveys` object (i.e., containing the information to determine the geometry a drillhole) to the plotter.

        If not already constructed, the mesh for the surveys will be created by the `drilldown.Surveys` object and added.

        Parameters
        ----------
        surveys : drilldown.Surveys
            The surveys (i.e., containing the azimuth, dip, and depth of each hole) to add to the plotter.

        opacity : float, optional
            The opacity of the surveys. Defaults to 1.

        *args : tuple
            Positional arguments to pass to `add_mesh()`.

        **kwargs : dict
            Keyword arguments to pass to `add_mesh()`.

        Returns
        -------
        pyvista.Actor
            The actor representing the surveys.

        Examples
        --------
        Create a `drilldown.Surveys` object and add it to the plotter:

        >>> import drilldown as dd
        >>> data_dict = dd.examples.load_tom_zone_macpass_project()
        >>> collar_data = data_dict["collar"]
        >>> survey_data = data_dict["survey"]
        >>> collars = dd.Collars()
        >>> collars.add_data(collar_data["hole_ID"], collar_data[["x", "y", "z"]])
        >>> surveys = dd.Surveys()
        >>> surveys.add_data(survey_data["hole_ID"], survey_data["depth"], survey_data["azimuth"], survey_data["dip"])
        >>> surveys.locate(collars)
        >>> pl = dd.Plotter()
        >>> pl.add_surveys(surveys)
        >>> pl.show()

        """

        from .holes import Surveys

        if not isinstance(surveys, Surveys):
            raise TypeError("surveys must be a DrillDown.Surveys object.")

        self.surveys = surveys
        if surveys.mesh is None:
            surveys.make_mesh()

        mesh = surveys.mesh

        mesh = mesh.tube(radius=0.2, n_sides=10, capping=True)
        actor = self.add_mesh(
            mesh, name="surveys", opacity=opacity, color="black", *args, **kwargs
        )
        self.survey_actor = actor

        return actor

    def add_intervals(
        self,
        intervals,
        name,
        opacity=1,
        selectable=True,
        radius=1.5,
        n_sides=20,
        capping=True,
        active_array_name=None,
        cmap=None,
        clim=None,
        selection_color="magenta",
        filter_opacity=0.1,
        accelerated_selection=False,
        visibility=True,
        *args,
        **kwargs,
    ):
        """Add a `drilldown.Intervals` object (i.e., containing drillhole data whose spatial extent is defined by from-to intervals) to the plotter as a data layer.

        If not already constructed, the mesh for the intervals will be created by the `drilldown.Intervals` object and added.

        Parameters
        ----------
        intervals : drilldown.Intervals
            The intervals to add to the plotter.

        name : str
            The name of the intervals data layer.

        opacity : float, optional
            The opacity of the intervals. Defaults to 1.

        selectable : bool, optional
            Whether the intervals are selectable with a cursor. Defaults to True.

        radius : float, optional
            The radius of the tube representing the intervals. Defaults to 1.5.

        n_sides : int, optional
            The number of sides of the tube representing the intervals. Defaults to 20.

        capping : bool, optional
            Whether to cap the ends of the tube segments representing the intervals. Defaults to True.

        active_array_name : str, optional
            The name of the array to use as the active array (i.e., the array by which the intervals are colored). Defaults to None.

        cmap : str, optional
            The name of the color map to use for coloring the intervals by the active array. Must be a `matplotlib` color map. Defaults to None.

        clim : tuple, optional
            The color map limits for the active array (i.e., specifying the range of the active array that the color map is distributed over). Use the format `(min, max)`. Defaults to None.


        selection_color : str, optional
            The color of the intervals when selected. Colors in any format accepted by `PyVista` are accepted. Defaults to "magenta".

        filter_opacity : float, optional
            When a filter is applied, the opacity of the filtered-out portion of the intervals. Defaults to 0.1.

        accelerated_selection : bool, optional
            Cursor selection of intervals will be accelerated if True, though with lower precision picking. Defaults to False.

        visibility : bool, optional
            Whether the intervals are visible. Defaults to True.

        *args : tuple
            Positional arguments to pass to `add_mesh()`.

        **kwargs : dict
            Keyword arguments to pass to `add_mesh()`.

        Returns
        -------
        pyvista.Actor
            The actor representing the intervals.

        Example
        -------
        Create a `drilldown.Intervals` object and add it to the plotter:

        >>> import drilldown as dd
        >>> data_dict = dd.examples.load_tom_zone_macpass_project()
        >>> collar_data = data_dict["collar"]
        >>> survey_data = data_dict["survey"]
        >>> assay_data = data_dict["assay"]
        >>> collars = dd.Collars()
        >>> collars.add_data(collar_data["hole_ID"], collar_data[["x", "y", "z"]])
        >>> surveys = dd.Surveys()
        >>> surveys.add_data(survey_data["hole_ID"], survey_data["depth"], survey_data["azimuth"], survey_data["dip"])
        >>> surveys.locate(collars)
        >>> assays = dd.Intervals()
        >>> assays.add_data(assay_data["hole_ID"], assay_data[["depth_from", "depth_to"]], assay_data)
        >>> assays.desurvey(surveys)
        >>> pl = dd.Plotter()
        >>> pl.add_intervals(assays, "assays")
        >>> pl.show()

        """

        from .holes import Intervals

        if not isinstance(intervals, Intervals):
            raise TypeError("intervals must be a DrillDown.Intervals object.")

        if intervals.mesh is None:
            intervals.make_mesh()

        mesh = intervals.mesh
        mesh = mesh.tube(radius=radius, n_sides=n_sides, capping=capping)

        kwargs = {k: v for k, v in kwargs.items() if k != "visibility"}
        actor = self.add_mesh(
            mesh,
            name=name,
            opacity=opacity,
            pickable=selectable,
            scalars=active_array_name,
            cmap=cmap,
            clim=clim,
            show_scalar_bar=False,
            *args,
            **kwargs,
        )

        actor.visibility = visibility

        cells_per_interval = n_sides
        if capping == True:
            cells_per_interval += 2

        layer = IntervalDataLayer(
            name,
            mesh,
            actor,
            self,
            visibility=visibility,
            opacity=opacity,
            cells_per_interval=cells_per_interval,
            selection_color=selection_color,
        )

        # handle categorical, continuous, and image arrays
        layer._categorical_array_names = intervals.categorical_array_names
        layer._continuous_array_names = intervals.continuous_array_names
        layer._image_array_names = intervals.image_array_names

        # enable decoding of hole IDs and categorical arrays
        layer.code_to_cat_map = intervals.code_to_cat_map

        # handle categorical color maps
        intervals._construct_categorical_cmap()
        layer.cat_to_color_map = intervals.cat_to_color_map
        layer.matplotlib_formatted_color_maps = (
            intervals.matplotlib_formatted_color_maps
        )

        self.layers.append(layer)

        return actor

    def add_points(
        self,
        points,
        name,
        opacity=1,
        selectable=True,
        point_size=10,
        active_array_name=None,
        cmap=None,
        clim=None,
        selection_color="magenta",
        filter_opacity=0.1,
        accelerated_selection=False,
        visibility=True,
        *args,
        **kwargs,
    ):
        """Add a `drilldown.Points` object (i.e., containing drillhole data defined as points) to the plotter as a data layer.

        If not already constructed, the mesh for the points will be created by the `drilldown.Points` object and added.

        Parameters
        ----------
        points : drilldown.Points
            The points to add to the plotter.

        name : str
            The name of the points data layer.

        opacity : float, optional
            The opacity of the points. Defaults to 1.

        selectable : bool, optional
            Whether the points are selectable with a cursor. Defaults to True.

        point_size : int or float, optional
            The size of the points. Defaults to 10.

        active_array_name : str, optional
            The name of the array to use as the active array (i.e., the array by which the points are colored). Defaults to None.

        cmap : str, optional
            The name of the color map to use for coloring the points by the active array. Must be a `matplotlib` color map. Defaults to None.

        clim : tuple, optional
            The color map limits for the active array (i.e., specifying the range of the active array that the color map is distributed over). Use the format `(min, max)`. Defaults to None.

        selection_color : str, optional
            The color of the points when selected. Colors in any format accepted by `PyVista` are accepted. Defaults to "magenta".

        filter_opacity : float, optional
            When a filter is applied, the opacity of the filtered-out portion of the points. Defaults to 0.1.

        accelerated_selection : bool, optional
            Cursor selection of points will be accelerated if True, though with lower precision picking. Defaults to False.

        visibility : bool, optional
            Whether the points are visible. Defaults to True.

        *args : tuple
            Positional arguments to pass to `add_mesh()`.

        **kwargs : dict
            Keyword arguments to pass to `add_mesh()`.

        Returns
        -------
        pyvista.Actor
            The actor representing the points.

        """
        from .holes import Points

        if not isinstance(points, Points):
            raise TypeError("points must be a DrillDown.Points object.")

        if points.mesh is None:
            points.make_mesh()

        mesh = points.mesh

        kwargs = {k: v for k, v in kwargs.items() if k != "visibility"}
        actor = self.add_mesh(
            mesh,
            name=name,
            opacity=opacity,
            pickable=selectable,
            point_size=point_size,
            render_points_as_spheres=True,
            scalars=active_array_name,
            show_scalar_bar=False,
            cmap=cmap,
            clim=clim,
            *args,
            **kwargs,
        )

        actor.visibility = visibility

        layer = PointDataLayer(
            name,
            mesh,
            actor,
            self,
            visibility=visibility,
            opacity=opacity,
            selection_color=selection_color,
        )

        # handle categorical, continuous, and image arrays
        layer._categorical_array_names = points.categorical_array_names
        layer._continuous_array_names = points.continuous_array_names
        layer._image_array_names = points.image_array_names

        # enable decoding of hole IDs and categorical arrays
        layer.code_to_cat_map = points.code_to_cat_map

        # handle categorical color maps
        points._construct_categorical_cmap()
        layer.cat_to_color_map = points.cat_to_color_map
        layer.matplotlib_formatted_color_maps = points.matplotlib_formatted_color_maps

        self.layers.append(layer)

        return actor

    def _on_single_click(self, *args):
        """Handle single click event."""
        picked_actor = self._pick()
        if picked_actor is not None:
            self._pick_on_single_click(picked_actor)

    def _on_double_click(self, *args):
        """Handle double click event."""
        picked_actor = self._pick()
        if picked_actor is not None:
            self._pick_on_dbl_click(picked_actor)

        elif len(self._active_selections_and_filters) > 0:
            selection_or_filter = self._active_selections_and_filters[-1]
            layer_name = list(selection_or_filter.keys())[0]
            layer = self.layers[layer_name]
            if selection_or_filter[layer_name] == "selection":
                layer._reset_selection()
            elif selection_or_filter[layer_name] == "filter":
                layer._reset_filter()

    def _pick(self):
        """Pick an actor at the current click position."""
        pos = self.click_position
        actor_picker = self.actor_picker
        actor_picker.Pick(pos[0], pos[1], 0, self.renderer)
        picked_actor = actor_picker.GetActor()

        return picked_actor

    def _pick_on_single_click(self, picked_actor):
        """Handle single click event on picked actor."""
        pos = self.click_position
        # make non-picked actors not pickable to improve performance
        self.actors_to_make_not_pickable_picker.Pick(pos[0], pos[1], 0, self.renderer)
        actors_to_make_not_pickable = actors_collection_to_list(
            self.actors_to_make_not_pickable_picker.GetActors()
        )
        actors_to_make_not_pickable.remove(picked_actor)

        prev_pickable = []
        for actor in actors_to_make_not_pickable:
            prev_pickable.append(actor.GetPickable())
            actor.SetPickable(False)

        # make selection
        for layer in self.layers:
            if (layer.actor == picked_actor) or (layer.filter_actor == picked_actor):
                layer._make_selection_by_pick(pos, picked_actor)
                self.picked_layer = layer

        # restore previous pickable state
        for actor, val in zip(actors_to_make_not_pickable, prev_pickable):
            actor.SetPickable(val)

    def _pick_on_dbl_click(self, picked_actor):
        """Handle double click event on picked actor."""
        pos = self.click_position
        # make non-picked actors not pickable to improve performance
        self.actors_to_make_not_pickable_picker.Pick(pos[0], pos[1], 0, self.renderer)
        actors_to_make_not_pickable = actors_collection_to_list(
            self.actors_to_make_not_pickable_picker.GetActors()
        )
        actors_to_make_not_pickable.remove(picked_actor)

        prev_pickable = []
        for actor in actors_to_make_not_pickable:
            prev_pickable.append(actor.GetPickable())
            actor.SetPickable(False)

        # make selection
        for layer in self.layers:
            if (layer.actor == picked_actor) or (layer.filter_actor == picked_actor):
                layer._make_selection_by_dbl_click_pick(pos, picked_actor)
                self.picked_layer = layer

        # restore previous pickable state
        for actor, val in zip(actors_to_make_not_pickable, prev_pickable):
            actor.SetPickable(val)

    def _reset_data(self, *args):
        """Reset selections and filters for all layers."""
        for layer in self.layers:
            layer._reset_selection()
            layer._reset_filter()

    @property
    def collar_visibility(self):
        """Return the visibility of the collars

        Returns
        -------
        bool
            Whether the collars are visible

        """
        return self._collar_visibility

    @collar_visibility.setter
    def collar_visibility(self, value):
        """Set the visibility of the collars

        Parameters
        ----------
        value : bool
            Whether the collars are visible

        """
        self.collar_actor.visibility = value
        self.render()

        self._collar_visibility = value

    @property
    def collar_opacity(self):
        """Return the opacity of the collars

        Returns
        -------
        float, int
            The opacity of the collars

        """
        return self._collar_opacity

    @collar_opacity.setter
    def collar_opacity(self, value):
        """Set the opacity of the collars

        Parameters
        ----------
        value : float, int
            The opacity of the collars, must be between 0 and 1

        """
        if value < 0 or value > 1:
            raise ValueError("opacity must be between 0 and 1")

        self.collar_actor.prop.opacity = value
        self.render()

        self._collar_opacity = value

    @property
    def collar_label_visibility(self):
        """Return the visibility of the collar labels

        Returns
        -------
        bool
            Whether the collar labels are visible

        """
        return self._collar_label_visibility

    @collar_label_visibility.setter
    def collar_label_visibility(self, value):
        """Set the visibility of the collar labels

        Parameters
        ----------
        value : bool
            Whether the collar labels are visible

        """
        if self.collar_label_actor is not None:
            self.collar_label_actor.SetVisibility(value)

        else:
            collar_mesh = self.collar_actor.mapper.dataset
            label_actor = self.add_point_labels(
                collar_mesh,
                collar_mesh["hole ID"],
                shape_opacity=0.5,
                show_points=False,
            )
            self.collar_label_actor = label_actor
            self._collar_label_visibility = value

        self.render()

        self._collar_label_visibility = value

    @property
    def survey_visibility(self):
        """Return the visibility of the surveys

        Returns
        -------
        bool
            Whether the surveys are visible

        """
        return self._survey_visibility

    @survey_visibility.setter
    def survey_visibility(self, value):
        """Set the visibility of the surveys

        Parameters
        ----------
        value : bool
            Whether the surveys are visible

        """
        self.survey_actor.visibility = value
        self.render()

        self._survey_visibility = value

    @property
    def survey_opacity(self):
        """Return the opacity of the surveys

        Returns
        -------
        float, int
            The opacity of the surveys

        """
        return self._survey_opacity

    @survey_opacity.setter
    def survey_opacity(self, value):
        """Set the opacity of the surveys

        Parameters
        ----------
        value : float, int
            The opacity of the surveys, must be between 0 and 1

        """
        if value < 0 or value > 1:
            raise ValueError("opacity must be between 0 and 1")

        self.survey_actor.prop.opacity = value
        self.render()

        self._survey_opacity = value

    def iframe(self, w="100%", h=None):
        """Return an iframe containing the plotter."""
        if h is None:
            h = self.height

        self._pv_viewer = show_trame(self, mode="server")
        self._trame_viewer = self._pv_viewer.viewer
        self._src = self._pv_viewer.src
        self._server = self._trame_viewer.server
        self._state = self._server.state
        self._iframe = IFrame(self._src, w, h)

        return self._iframe

    def _get_server(self):
        """Return the plotter's trame server."""
        pv_viewer = show_trame(self, mode="server")
        trame_viewer = pv_viewer.viewer
        server = trame_viewer.server

        return server

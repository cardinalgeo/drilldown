from vtk import (
    vtkPicker,
    vtkPropPicker,
    vtkMapper,
)
from pyvista import Plotter
from pyvista.trame.jupyter import show_trame
from trame.app import get_server

from IPython.display import IFrame
import uuid

from .layer.layer import IntervalDataLayer, PointDataLayer
from .layer.layer_list import LayerList
from .utils import is_jupyter


def is_numeric_tuple(tup):
    return all(isinstance(x, (int, float)) for x in tup)


def actors_collection_to_list(actors_collection):
    actors_collection.InitTraversal()
    actors_list = []
    for i in range(actors_collection.GetNumberOfItems()):
        actors_list.append(actors_collection.GetNextActor())

    return actors_list


class DrillDownPlotter(Plotter):
    """Plotting object for displaying drillholes and related datasets."""

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
        self.survey_actor = None

        self._show_collar_labels = True

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

    def add_collars(self, collars, show_labels=True, opacity=1, *args, **kwargs):
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
            point_size=15,
            pickable=False,
            *args,
            **kwargs,
        )
        self.collar_actor = actor

        if show_labels == True:
            label_actor = self.add_point_labels(
                mesh, mesh["hole ID"], shape_opacity=0.5, show_points=False
            )
            self.collar_label_actor = label_actor

            return actor, label_actor

        else:
            return actor

    def add_surveys(self, surveys, opacity=1, *args, **kwargs):
        from .holes import Surveys

        if not isinstance(surveys, Surveys):
            raise TypeError("surveys must be a DrillDown.Surveys object.")

        self.surveys = surveys
        if surveys.mesh is None:
            surveys.make_mesh()

        mesh = surveys.mesh
        actor = self.add_mesh(mesh, name="surveys", opacity=opacity, *args, **kwargs)
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
        from .holes import Intervals

        if not isinstance(intervals, Intervals):
            raise TypeError("intervals must be a DrillDown.Intervals object.")

        if intervals.mesh is None:
            intervals.make_mesh()

        mesh = intervals.mesh
        mesh = mesh.tube(radius=radius, n_sides=n_sides, capping=False)

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

        layer = IntervalDataLayer(
            name,
            mesh,
            actor,
            self,
            visibility=visibility,
            opacity=opacity,
            n_sides=n_sides,
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
        picked_actor = self._pick()
        if picked_actor is not None:
            self._pick_on_single_click(picked_actor)

    def _on_double_click(self, *args):
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
        pos = self.click_position
        actor_picker = self.actor_picker
        actor_picker.Pick(pos[0], pos[1], 0, self.renderer)
        picked_actor = actor_picker.GetActor()

        return picked_actor

    def _pick_on_single_click(self, picked_actor):
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
        for layer in self.layers:
            layer._reset_selection()
            layer._reset_filter()

    def iframe(self, w="100%", h=None):
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
        pv_viewer = show_trame(self, mode="server")
        trame_viewer = pv_viewer.viewer
        server = trame_viewer.server

        return server

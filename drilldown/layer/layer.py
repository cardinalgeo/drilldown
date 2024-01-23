from typing import Any


class BaseLayer:
    def __init__(
        self, name, mesh, actor, plotter, visible=True, opacity=1, pickable=True
    ):
        self.name = name
        self.mesh = mesh
        self.actor = actor
        self._visible = visible
        self._opacity = opacity
        self.pickable = pickable
        self.plotter = plotter

    @property
    def visible(self):
        return self._visible

    @visible.setter
    def visible(self, value):
        if value == True:
            self.actor.prop.opacity = 1
        elif value == False:
            self.actor.prop.opacity = 0
        else:
            raise ValueError("visible must be True or False")

        self._visible = value

        self.plotter.render()

    @property
    def opacity(self):
        return self._opacity

    @opacity.setter
    def opacity(self, value):
        if value < 0 or value > 1:
            raise ValueError("opacity must be between 0 and 1")

        self.actor.prop.opacity = value

        self._opacity = value

        self.plotter.render()


class DataLayer(BaseLayer):
    def __init__(
        self,
        name,
        mesh,
        actor,
        plotter,
        visible=True,
        opacity=1,
        pickable=True,
        active_var=None,
        cmap=None,
        clim=None,
    ):
        super().__init__(name, mesh, actor, plotter, visible, opacity, pickable)

        self._active_var = active_var
        self._cmap = cmap
        self._clim = clim

        self._continuous_array_names = []
        self._categorical_array_names = []

        self._selection_actor = None
        self._filter_actor = None

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

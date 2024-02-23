from trame.widgets import vuetify

from ..layer.layer import IntervalDataLayer, PointDataLayer


class ControlsUI(vuetify.VCard):
    def __init__(self, active_layer):
        super().__init__(
            elevation=0,
            style="background-color: #f5f5f5; overflow: auto; height: 40%; width: 90%; margin-left: auto; margin-right: auto; margin-top: auto; margin-bottom: auto;",
        )

        with self:
            vuetify.VSlider(
                hide_details=True,
                label="opacity",
                v_model=("opacity",),
                max=1,
                min=0,
                step=0.001,
                style="width: 90%; margin-left: auto; margin-right: auto",
            )
            if (isinstance(active_layer, IntervalDataLayer)) or (
                isinstance(active_layer, PointDataLayer)
            ):
                vuetify.VDivider(
                    classes="mb-2",
                    v_show=("divider_visible", True),
                    style="width: 90%; margin-left: auto; margin-right: auto",
                )
                visible = True
            else:
                visible = False

            vuetify.VSelect(
                label="active array name",
                v_show=("active_array_name_visible", visible),
                v_model=(
                    "active_array_name",
                    active_layer.active_array_name,
                ),
                items=("array_names",),
                classes="pt-1",
                style="width: 90%; margin-left: auto; margin-right: auto",
                # **DROPDOWN_STYLES,
            )

            if (visible == False) or (
                active_layer.active_array_name in active_layer.categorical_array_names
            ):
                visible = False

            vuetify.VSelect(
                label="colormap",
                v_show=("cmap_visible", visible),
                v_model=("cmap",),
                items=("cmap_fields",),
                classes="pt-1",
                style="width: 90%; margin-left: auto; margin-right: auto",
            )

            vuetify.VRangeSlider(
                label="colormap limits",
                v_show=("clim_visible", visible),
                v_model=("clim",),
                min=("clim_min",),
                max=("clim_max",),
                step=("clim_step",),
                classes="pt-1",
                style="width: 90%; margin-left: auto; margin-right: auto",
            )

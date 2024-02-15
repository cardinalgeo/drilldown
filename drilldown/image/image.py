from trame.app import get_server
from trame.widgets import vuetify, leaflet
from trame.ui.vuetify import SinglePageLayout
from pyvista.trame.jupyter import elegantly_launch
from trame_server.utils.browser import open_browser
import io
from aiohttp import web
import large_image
import uuid

from ..utils import is_jupyter


class ImageViewer:
    def __init__(self, name=None, server=None, *args, **kwargs):
        if name is not None:
            self.name = name
        else:
            self.name = str(uuid.uuid4())

        if server is not None:
            self.server = server
        else:
            self.server = get_server(self.name)

        self.state = self.server.state
        self.ctrl = self.server.controller
        self.server.client_type = "vue2"

        self.state.carousel = False

        self.image_handler = ImageHandler(self.name)
        self.ctrl.on_server_bind.add(self.image_handler.add_endpoint)
        self._ui = self._initialize_ui()

        self.image_filenames = None
        self.active_image_filename = None

        self.carousel_position = 0

        # control auto updating
        self.auto_update = True

    def show(self, image_filenames=None, inline=True):
        if image_filenames is None:
            if self.image_filenames is not None:
                image_filenames = self.image_filenames

                if isinstance(image_filenames, list):
                    if len(image_filenames) == 1:
                        self.update(image_filenames[0])
                    else:
                        self.state.carousel = True
                else:
                    self.update(image_filenames)
            else:
                raise ValueError("No image filename(s) provided.")

            self.image_filenames = image_filenames
            self.active_image_filename = image_filenames[0]

        if inline == True:
            if is_jupyter():
                elegantly_launch(self.server)  # launch server in nb w/o using await

                return self._ui

            else:
                raise ValueError("Inline mode only available in Jupyter notebook.")

        else:
            if is_jupyter():
                self.server.start(
                    exec_mode="task",
                    port=0,
                    open_browser=True,
                )
            else:
                self.server.start(
                    port=0,
                    open_browser=True,
                )

            ctrl = self.server.controller
            ctrl.on_server_ready.add(lambda **kwargs: open_browser(self.server))

    def _initialize_ui(self):
        with SinglePageLayout(self.server, template_name=self.name) as layout:
            layout.toolbar.hide()
            layout.footer.hide()
            with layout.content:
                with vuetify.VCard(
                    style="position: absolute; top: 250px; left: 20px; z-index: 1000; height: 36px;"
                ):
                    with vuetify.VBtn(
                        icon=True, v_show="carousel", click=self.carousel_left
                    ):
                        vuetify.VIcon("mdi-arrow-left")
                with vuetify.VCard(
                    style="position: absolute; top: 250px; right: 20px; z-index: 1000; height: 36px;"
                ):
                    with vuetify.VBtn(
                        icon=True, v_show="carousel", click=self.carousel_right
                    ):
                        vuetify.VIcon("mdi-arrow-right")
                with vuetify.VContainer(
                    fluid=True, classes="fill-height pa-0 ma-0", style="z-index: 1"
                ):
                    with leaflet.LMap(zoom=("zoom", 1), max_zoom=100, min_zoom=1):
                        # tiles
                        leaflet.LTileLayer(
                            url=("tile_url", ""),
                            no_wrap=True,
                            options=("{ maxNativeZoom: 4 }",),
                        )

        return layout

    def update(self, filename):
        self.image_handler.open_image(filename)
        with self.state:
            self.state.tile_url = self.image_handler.create_request()

    def carousel_left(self):
        if self.carousel_position > 0:
            self.carousel_position -= 1
            self.update(self.image_filenames[self.carousel_position])
            self.active_image_filename = self.image_filenames[self.carousel_position]

    def carousel_right(self):
        if self.carousel_position < len(self.image_filenames) - 1:
            self.carousel_position += 1
            self.update(self.image_filenames[self.carousel_position])
            self.active_image_filename = self.image_filenames[self.carousel_position]


# Open raster with large-image
class ImageHandler:
    def __init__(self, name):
        self.name = name
        self.request_template = f"/{name}_tile/{{x}}/{{y}}/{{z}}.png"
        self.routes = [web.get(self.request_template, self.tile)]

    def open_image(self, image_path):
        self.image = image_path
        self.src = large_image.open(
            self.image,
            # os.path.join(self.image),
            # file_path,
            encoding="PNG",
            edge="#DDDDDD",
        )

    def restyle(self, style):
        self.src = large_image.open(
            self.image,
            # os.path.join(self.image)
            # file_path,
            encoding="PNG",
            edge="#DDDDDD",
            style=style,
        )

    async def tile(self, request):
        """REST endpoint to serve tiles from image in slippy maps standard."""
        self.z = int(request.match_info["z"])
        self.x = int(request.match_info["x"])
        self.y = int(request.match_info["y"])

        try:  # get around issue where coordinates within viewer are outside tile source
            tile_binary = self.src.getTile(self.x, self.y, self.z)
            return web.Response(body=io.BytesIO(tile_binary), content_type="image/png")
        except:
            pass

    def add_endpoint(self, wslink_server):
        """Add our custom REST endpoints to the trame server."""
        wslink_server.app.add_routes(self.routes)

    def create_request(self, cmap=None, threshold_min=None, threshold_max=None):
        request = self.request_template + f"?filename={self.image}"
        if cmap is not None:
            request += f"&cmap={cmap}"
        if threshold_min is not None:
            request += f"&threshold_min={threshold_min}"
        if threshold_max is not None:
            request += f"&threshold_max={threshold_max}"

        return request

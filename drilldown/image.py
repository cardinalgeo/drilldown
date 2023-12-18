import large_image
from large_image.tilesource.jupyter import launch_tile_server
from ipyleaflet import TileLayer, Map
from ipywidgets import Layout
import os


# Open raster with large-image
class ImageHandler:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def open_image(self, image):
        self.image = image
        src = large_image.open(
            os.path.join(self.folder_path, self.image),
            # file_path,
            encoding="PNG",
            edge="#DDDDDD",
        )
        self.src = src
        if src._jupyter_server_manager is None:
            # Must relaunch to ensure style updates work
            src._jupyter_server_manager = launch_tile_server(src)
        else:
            # Must update the source on the manager incase the previous reference is bad
            src._jupyter_server_manager.tile_source = src

        port = src._jupyter_server_manager.port

        if src.JUPYTER_PROXY:
            if isinstance(src.JUPYTER_PROXY, str):
                base_url = f'{src.JUPYTER_PROXY.rstrip("/")}/{port}'
            else:
                base_url = f"/proxy/{port}"
        else:
            base_url = f"http://{src.JUPYTER_HOST}:{port}"

        self.base_url = base_url

        # Use repr in URL params to prevent caching across sources/styles
        endpoint = f"tile?z={{z}}&x={{x}}&y={{y}}&encoding=png&repr={src.__repr__()}"
        self.endpoint = endpoint

        self.metadata = src.getMetadata()


class ImageViewer:
    def __init__(self):
        pass

    def set_directory_path(self, directory_path):
        self.directory_path = directory_path
        self.image_handler = ImageHandler(directory_path)

    def _create_tile_layer(self):
        handler = self.image_handler
        layer = TileLayer(
            url=f"{handler.base_url}/{handler.endpoint}",
            # attribution='Tiles served with large-image',
            min_zoom=0,
            max_native_zoom=handler.metadata["levels"],
            max_zoom=handler.metadata["levels"] + 1,
            tile_size=handler.metadata["tileWidth"],
        )
        self.tile_layer = layer
        return layer

    def _create_crs(self):
        handler = self.image_handler
        metadata = handler.metadata
        crs = dict(
            name="PixelSpace",
            custom=True,
            # Why does this need to be 256?
            resolutions=[256 * 2 ** (-l) for l in range(20)],
            # This works but has x and y reversed
            proj4def="+proj=longlat +axis=esu",
            bounds=[[0, 0], [metadata["sizeY"], metadata["sizeX"]]],
            origin=[0, 0],
            # This almost works to fix the x, y reversal, but
            # - bounds are weird and other issues occur
            # proj4def='+proj=longlat +axis=seu',
            # bounds=[[-metadata['sizeX'],-metadata['sizeY']],[metadata['sizeX'],metadata['sizeY']]],
            # origin=[0,0],
        )
        self.crs = crs
        return crs

    def _create_viewer(self):
        handler = self.image_handler
        src = handler.src
        metadata = handler.metadata

        try:
            default_zoom = metadata["levels"] - metadata["sourceLevels"]
        except KeyError:
            default_zoom = 0

        m = Map(
            crs=self.crs,
            basemap=self.tile_layer,
            center=src.getCenter(srs="EPSG:4326"),
            zoom=default_zoom,
            max_zoom=metadata["levels"] + 1,
            min_zoom=0,
            scroll_wheel_zoom=True,
            dragging=True,
            layout=Layout(height="800px")
            # attribution_control=False,
        )
        self.viewer = m

        return m

    def view_image(self, filename):
        self.image_handler.open_image(filename)
        self._create_tile_layer()
        self._create_crs()
        self._create_viewer()
        return self.viewer

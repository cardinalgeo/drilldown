class ImageMixin:
    def __init__(self):
        pass

    def selected_image(self, array_name=None):
        from .image import ImageViewer

        data = self.selected_data
        if data.shape[0] != 1:
            raise ValueError(
                "More than one interval or point selected. Please select only one."
            )

        if array_name is None:
            image_array_names = self._image_array_names
            if len(image_array_names) == 1:
                array_name = image_array_names[0]
            else:
                raise ValueError(
                    "Multiple image arrays present. Please specify array name."
                )

        image_filename = data[array_name].values[0]
        if image_filename != "nan":
            im_viewer = ImageViewer()
            im_viewer.image_filename = image_filename

            return im_viewer

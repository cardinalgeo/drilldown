class ImageMixin:
    def _make_single_selection(self, picked, on_filter=False):
        super(ImageMixin, self)._make_single_selection(picked, on_filter=on_filter)

        if (self.im_viewer is not None) and (self.im_viewer.auto_update == True):
            data = self.selected_data
            image_array_names = self._image_array_names
            if len(image_array_names) == 1:
                array_name = image_array_names[0]
            else:
                raise ValueError(
                    "Multiple image arrays present. Please specify array name."
                )

            image_filename = data[array_name].values[0]
            self.im_viewer.update(image_filename)

    def selected_image(self, array_name=None, auto_update=True):
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
            im_viewer.auto_update = auto_update
            self.im_viewer = im_viewer

            im_viewer.image_filename = image_filename

            return im_viewer

class ImageMixin:
    def _update_selection_object(self):
        super(ImageMixin, self)._update_selection_object()

        if (self.im_viewer is not None) and (self.im_viewer.auto_update == True):
            data = self.selected_data
            if self.image_viewer_active_array_name is None:
                image_array_names = self._image_array_names
            else:  # use the array name that was last used
                image_array_names = self.image_viewer_active_array_name

            if isinstance(image_array_names, list):
                if len(image_array_names) == 1:
                    array_name = image_array_names[0]
                else:
                    raise ValueError(
                        "Multiple image arrays present. Please specify array name."
                    )
            else:
                array_name = image_array_names

            image_filenames = data[array_name].values.tolist()
            if len(image_filenames) != 0:
                image_filename = image_filenames[0]
                self.im_viewer.update(image_filename)
                if len(image_filenames) > 1:
                    self.im_viewer.state.carousel = True
                else:
                    self.im_viewer.state.carousel = False

                self.im_viewer.image_filenames = image_filenames

    def selected_images(self, array_name=None, auto_update=True):
        from .image import ImageViewer

        self.image_viewer_active_array_name = array_name
        if array_name is None:
            image_array_names = self._image_array_names
            if len(image_array_names) == 1:
                array_name = image_array_names[0]
            else:
                raise ValueError(
                    "Multiple image arrays present. Please specify array name."
                )

        if array_name not in self.array_names:  # check if array name is valid
            raise ValueError(f"{array_name} is not a valid array name.")

        data = self.selected_data

        image_filenames = data[array_name].to_list()
        if len(image_filenames) != 0:
            im_viewer = ImageViewer()
            im_viewer.auto_update = auto_update
            self.im_viewer = im_viewer

            im_viewer.image_filenames = image_filenames

            return im_viewer

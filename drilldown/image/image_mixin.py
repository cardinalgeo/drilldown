class ImageMixin:
    def __init__(self):
        pass

    def selected_image(self, var_name=None, **kwargs):
        from .image import ImageViewer

        data = self.selected_data()
        if data.shape[0] != 1:
            raise ValueError(
                "More than one interval or point selected. Please select only one."
            )

        if var_name is None:
            dataset_name = self.selection_actor_name.split(" ")[0]
            dataset = self.datasets[dataset_name]
            image_var_names = dataset.image_var_names
            if len(image_var_names) == 1:
                var_name = image_var_names[0]
            else:
                raise ValueError(
                    "Multiple image variables present. Please specify variable name."
                )

        image_filename = data[var_name].values[0]
        im_viewer = ImageViewer()
        im_viewer.image_filename = image_filename

        return im_viewer

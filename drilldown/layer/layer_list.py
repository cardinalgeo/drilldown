class LayerList(list):
    """List-like layer collection"""

    def __getitem__(self, value):
        if isinstance(value, str):
            for layer in self:
                if layer.name == value:
                    return layer

            return None

        else:
            return super().__getitem__(value)

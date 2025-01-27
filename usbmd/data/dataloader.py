import keras

from usbmd.backend.tensorflow.dataloader import H5Generator, Resizer
from usbmd.utils import translate


class H5Dataloader(H5Generator):
    def __init__(
        self,
        resize_type: str = "center_crop",
        image_size: tuple | None = None,
        image_range: tuple = (0, 255),
        normalization_range: tuple = (0, 1),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.resize_type = resize_type
        self.image_size = image_size
        self.image_range = image_range
        self.normalization_range = normalization_range
        self.resizer = Resizer(
            resize_type=resize_type, image_size=image_size, keras=keras
        )

    def __getitem__(self, index):
        out = super().__getitem__(index)
        if self.return_filename:
            images, filenames = out
        else:
            images = out

        images = translate(images, self.image_range, self.normalization_range)

        if self.image_size is not None:
            images = self.resizer(images)

        if self.return_filename:
            return images, filenames
        else:
            return images

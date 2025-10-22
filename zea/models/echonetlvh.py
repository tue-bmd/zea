"""EchoNetLVH model for segmentation of PLAX view cardiac ultrasound. For more details see https://echonet.github.io/lvh/index.html."""

import numpy as np
from keras import ops

from zea.internal.registry import model_registry
from zea.models.base import BaseModel
from zea.models.deeplabv3 import DeeplabV3Plus
from zea.models.preset_utils import register_presets
from zea.models.presets import echonet_lvh_presets
from zea.utils import translate


@model_registry(name="echonetlvh")
class EchoNetLVH(BaseModel):
    """
    EchoNet Left Ventricular Hypertrophy (LVH) model for echocardiogram analysis.

    This model performs semantic segmentation on echocardiogram images to identify
    key anatomical landmarks for measuring left ventricular wall thickness:
    - LVPWd_1: Left Ventricular Posterior Wall point 1
    - LVPWd_2: Left Ventricular Posterior Wall point 2
    - IVSd_1: Interventricular Septum point 1
    - IVSd_2: Interventricular Septum point 2

    The model outputs 4-channel logits corresponding to heatmaps for each landmark.

    For more information, see the original project page at https://echonet.github.io/lvh/index.html
    """

    def __init__(self, **kwargs):
        """
        Initialize the EchoNetLVH model.

        Args:
            **kwargs: Additional keyword arguments passed to BaseModel
        """
        super().__init__(**kwargs)

        # Pre-computed coordinate grid for efficient processing
        self.coordinate_grid = ops.stack(
            ops.cast(ops.convert_to_tensor(np.indices((224, 224))), "float32"), axis=-1
        )

        # Initialize the underlying segmentation network
        self.network = DeeplabV3Plus(image_shape=(224, 224, 3), num_classes=4)

    def call(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (Tensor): Input images of shape [B, H, W, C]. They should
                be scan converted, with pixel values in range [0, 255].

        Returns:
            Tensor: Logits of shape [B, H, W, 4] with 4 channels for each landmark
        """
        assert len(ops.shape(inputs)) == 4

        # Store original dimensions for output resizing
        original_size = ops.shape(inputs)[1:3]

        # Resize to network input size
        inputs_resized = ops.image.resize(inputs, size=(224, 224))

        # Get network predictions
        logits = self.network(inputs_resized)

        # Resize logits back to original input dimensions
        logits_output = ops.image.resize(logits, original_size)
        return logits_output

    def extract_key_points_as_indices(self, logits):
        """
        Extract key point coordinates from logits using center-of-mass calculation.

        Args:
            logits (Tensor): Model output logits of shape [B, H, W, 4]

        Returns:
            Tensor: Key point coordinates of shape [B, 4, 2] where each point
                   is in (x, y) format
        """
        # Create coordinate grid for the current logit dimensions
        input_shape = ops.shape(logits)[1:3]
        input_space_coordinate_grid = ops.stack(
            ops.cast(ops.convert_to_tensor(np.indices(input_shape)), "float32"), axis=-1
        )

        # Transpose logits to [B, 4, H, W] for vectorized processing
        logits_batchified = ops.transpose(logits, (0, 3, 1, 2))

        # Extract expected coordinates for each channel
        return ops.flip(
            ops.vectorized_map(
                lambda logit: self.expected_coordinate(logit, input_space_coordinate_grid),
                logits_batchified,
            ),
            axis=-1,  # Flip to convert from (y, x) to (x, y)
        )

    def expected_coordinate(self, mask, coordinate_grid=None):
        """
        Compute the expected coordinate (center-of-mass) of a heatmap.

        This implements a differentiable version of taking the max of a heatmap
        by computing the weighted average of coordinates.

        Reference: https://arxiv.org/pdf/1711.08229

        Args:
            mask (Tensor): Heatmap of shape [B, H, W]
            coordinate_grid (Tensor, optional): Grid of coordinates. If None,
                                              uses self.coordinate_grid

        Returns:
            Tensor: Expected coordinates of shape [B, 2] in (x, y) format
        """
        if coordinate_grid is None:
            coordinate_grid = self.coordinate_grid

        # Ensure mask values are non-negative and normalized
        mask_clipped = ops.clip(mask, 0, None)
        mask_normed = mask_clipped / ops.max(mask_clipped)

        def safe_normalize(m):
            mask_sum = ops.sum(m)
            return ops.where(mask_sum > 0, m / mask_sum, m)

        coordinate_probabilities = ops.map(safe_normalize, mask_normed)

        # Add dimension for broadcasting with coordinate grid
        coordinate_probabilities = ops.expand_dims(coordinate_probabilities, axis=-1)

        # Compute weighted average of coordinates
        expected_coordinate = ops.sum(
            ops.expand_dims(coordinate_grid, axis=0) * coordinate_probabilities,
            axis=(1, 2),
        )

        # Flip to convert from (y, x) to (x, y) format for euclidean distance calculation
        return ops.flip(expected_coordinate, axis=-1)

    def overlay_labels_on_image(self, image, label, alpha=0.5):
        """
        Overlay predicted heatmaps and connecting lines on the input image.

        Args:
            image (Tensor): Input image of shape [H, W] or [H, W, C]
            label (Tensor): Predicted logits of shape [H, W, 4]
            alpha (float): Blending factor for overlay (0=transparent, 1=opaque)

        Returns:
            ndarray: Image with overlaid heatmaps and measurements of shape [H, W, 3]
        """
        try:
            import cv2

        except ImportError as exc:
            raise ImportError(
                "OpenCV is required for `EchoNetLVH.overlay_labels_on_image`. "
                "Please install it with 'pip install opencv-python' or "
                "'pip install opencv-python-headless'."
            ) from exc

        # Color scheme for each landmark
        overlay_colors = np.array(
            [
                [1, 1, 0],  # Yellow (LVPWd_X1)
                [1, 0, 1],  # Magenta (LVPWd_X2)
                [0, 1, 1],  # Cyan (IVSd_X1)
                [0, 1, 0],  # Green (IVSd_X2)
            ],
        )

        # Convert to numpy and ensure RGB format
        image = ops.convert_to_numpy(image)
        label = ops.convert_to_numpy(label)

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        else:
            image = image.copy()

        # Normalize each channel to [0, 1] for proper visualization
        label = np.clip(label, 0, None)
        for ch in range(label.shape[-1]):
            max_val = np.max(label[..., ch])
            if max_val > 0:
                label[..., ch] = label[..., ch] / max_val

        # Initialize overlay and tracking variables
        overlay = np.zeros_like(image, dtype=np.float32)
        centers = []

        # Process each landmark channel
        for ch in range(4):
            # Square the mask to enhance peak responses
            mask = label[..., ch] ** 2
            color = overlay_colors[ch]

            # Find center of mass for this channel
            center_coords = self.expected_coordinate(ops.expand_dims(mask, axis=0))
            center_x = ops.convert_to_numpy(center_coords[0, 0])
            center_y = ops.convert_to_numpy(center_coords[0, 1])

            # Bounds check before conversion to int
            if 0 <= center_x < image.shape[1] and 0 <= center_y < image.shape[0]:
                center = (int(center_x), int(center_y))
            else:
                center = None

            if center is not None:
                # Blend heatmap with overlay
                mask_alpha = mask * alpha
                for c in range(3):
                    overlay[..., c] += mask_alpha * color[c]
            centers.append(center)

        # Draw connecting lines between consecutive landmarks
        for i in range(3):
            pt1, pt2 = centers[i], centers[i + 1]
            if pt1 is not None and pt2 is not None:
                color = tuple(int(x) for x in overlay_colors[i])

                # Create line mask
                line_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.line(line_mask, pt1, pt2, color=1, thickness=2)

                # Apply line to overlay
                for c in range(3):
                    overlay[..., c][line_mask.astype(bool)] = color[c] * alpha

        # Blend overlay with original image
        overlay = np.clip(overlay, 0, 1)
        out = image.astype(np.float32)
        blend_mask = np.any(overlay > 0.02, axis=-1)
        out[blend_mask] = (1 - alpha) * out[blend_mask] + overlay[blend_mask]

        return np.clip(out, 0, 1)

    def visualize_logits(self, images, logits):
        """
        Create visualization of model predictions overlaid on input images.

        Args:
            images (Tensor): Input images of shape [B, H, W, C]
            logits (Tensor): Model predictions of shape [B, H, W, 4]

        Returns:
            Tensor: Images with overlaid predictions of shape [B, H, W, 3]
        """
        # Store original dimensions for final output
        original_size = ops.shape(images)[1:3]

        # Resize to standard processing size
        images_resized = ops.image.resize(images, size=(224, 224), interpolation="nearest")
        logits_resized = ops.image.resize(logits, size=(224, 224), interpolation="nearest")

        # Normalize images to [0, 1] range
        images_clipped = ops.clip(images_resized, 0, 255)
        images = translate(images_clipped, range_from=(0, 255), range_to=(0, 1))

        # Generate overlays for each image in the batch
        images_with_overlay = []
        for img, logit_heatmap in zip(images, logits_resized):
            overlay = self.overlay_labels_on_image(img, logit_heatmap)
            images_with_overlay.append(overlay)

        # Stack results and resize back to original dimensions
        images_with_overlay = np.stack(images_with_overlay, axis=0)
        return ops.image.resize(images_with_overlay, original_size)


# Register model presets
register_presets(echonet_lvh_presets, EchoNetLVH)

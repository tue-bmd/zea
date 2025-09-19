"""Model presets for zea.models"""

taesdxl_presets = {
    "taesdxl": {
        "metadata": {
            "description": "Tiny Autoencoder (TAESD) model",
            "params": 0,
            "path": "taesdxl",
        },
        "hf_handle": "hf://zeahub/taesdxl",
    },
}

taesdxl_encoder_presets = {
    "taesdxl_encoder": {
        "metadata": {
            "description": "Tiny encoder from TAESD model",
            "params": 0,
            "path": "taesdxl_encoder",
        },
        "hf_handle": "hf://zeahub/taesdxl",
    },
}

taesdxl_decoder_presets = {
    "taesdxl_decoder": {
        "metadata": {
            "description": "Tiny decoder from TAESD model",
            "params": 0,
            "path": "taesdxl_decoder",
        },
        "hf_handle": "hf://zeahub/taesdxl",
    },
}

echonet_dynamic_presets = {
    "echonet-dynamic": {
        "metadata": {
            "description": (
                "EchoNet-Dynamic segmentation model for cardiac ultrasound segmentation. "
                "Original paper and code: https://echonet.github.io/dynamic/"
            ),
            "params": 0,
            "path": "echonet",
        },
        "hf_handle": "hf://zeahub/echonet-dynamic",
    },
}

camus_presets = {
    "augmented_camus_seg": {
        "metadata": {
            "description": (
                "Augmented CAMUS segmentation model for cardiac ultrasound segmentation. "
                "Original paper and code: https://arxiv.org/abs/2502.20100"
            ),
            "params": 0,
            "path": "augmented_camus_seg",
        },
        "hf_handle": "hf://gillesvdv/augmented_camus_seg",
    },
}

echonet_lvh_presets = {
    "echonetlvh": {
        "metadata": {
            "description": (
                "EchoNetLVH segmentation model for PLAX-view cardiac ultrasound segmentation. "
                "Trained on images of size (224, 224)."
            ),
            "params": 0,
            "path": "echonetlvh",
        },
        "hf_handle": "hf://zeahub/echonetlvh",
    },
}

lpips_presets = {
    "lpips": {
        "metadata": {
            "description": "Learned Perceptual Image Patch Similarity (LPIPS) metric.",
            "params": 14716160,
            "path": "lpips",
        },
        "hf_handle": "hf://zeahub/lpips",
    },
}

unet_presets = {
    "unet-echonet-inpainter": {
        "metadata": {
            "description": (
                "U-Net model used to inpaint skipped lines (columns). "
                "Trained on 75% masked data (center values)."
            ),
            "params": 0,
            "path": "unet",
        },
        "hf_handle": "hf://zeahub/unet-echonet-inpainter",
    },
}

dense_presets = {}

diffusion_model_presets = {
    "diffusion-echonet-dynamic": {
        "metadata": {
            "description": ("Diffusion model trained on EchoNet-Dynamic dataset."),
            "params": 0,
            "path": "diffusion",
        },
        "hf_handle": "hf://zeahub/diffusion-echonet-dynamic",
    },
}

carotid_segmenter_presets = {
    "carotid-segmenter": {
        "metadata": {
            "description": (
                "Carotid segmentation model based on U-Net architecture. "
                "Trained on labeled simulated data and unlabeled invivo data."
            ),
            "params": 848461,
            "path": "carotid_segmenter",
        },
        "hf_handle": "hf://zeahub/carotid-segmenter",
    },
}

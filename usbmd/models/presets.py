"""Model presets"""

taesdxl_presets = {
    "taesdxl": {
        "metadata": {
            "description": "Tiny Autoencoder (TAESD) model",
            "params": 0,
            "path": "taesdxl",
        },
        "hf_handle": "hf://usbmd/taesdxl",
    },
}

taesdxl_encoder_presets = {
    "taesdxl_encoder": {
        "metadata": {
            "description": "Tiny encoder from TAESD model",
            "params": 0,
            "path": "taesdxl_encoder",
        },
        "hf_handle": "hf://usbmd/taesdxl",
    },
}

taesdxl_decoder_presets = {
    "taesdxl_decoder": {
        "metadata": {
            "description": "Tiny decoder from TAESD model",
            "params": 0,
            "path": "taesdxl_decoder",
        },
        "hf_handle": "hf://usbmd/taesdxl",
    },
}

echonet_segment_presets = {
    "echonet_segment": {
        "metadata": {
            "description": (
                "EchoNet-Dynamic segmentation model for cardiac ultrasound segmentation. "
                "Original paper and code: https://echonet.github.io/dynamic/"
            ),
            "params": 0,
            "path": "echonet",
        },
        "hf_handle": "hf://usbmd/taesdxl",
    },
}

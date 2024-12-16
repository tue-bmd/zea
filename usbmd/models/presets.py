"""Model presets"""

taesdxl_presets = {
    "taesdxl": {
        "metadata": {
            "description": ("Tiny Autoencoder (TAESD) model"),
            "params": 0,
            "path": "taesd",
        },
        "hf_handle": "hf://usbmd/taesdxl",
    },
}

echonet_presets = {
    "echonet": {
        "metadata": {
            "description": (
                "EchoNet-Dynamic segmentation model for cardiac ultrasound segmentation. "
                "Original paper and code: https://echonet.github.io/dynamic/"
            ),
            "params": 0,
            "path": "echonet",
        },
        "hf_handle": "hf://usbmd/echonet",
    },
}

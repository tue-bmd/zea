"""USBMD GUI app
Author(s): Tristan Stevens
Date: 19/04/2023

TODO:
- add schema, such that options are available
- add option picker entry
- more error handling
- fix lists
- better tickboxes
- labels prettier
- fix load config, such that are inserted in entries

"""
import warnings
from threading import Thread
from tkinter import ttk

import usbmd
from usbmd.common import set_data_paths
from usbmd.setup_usbmd import setup_config
from usbmd.utils.config import load_config_from_yaml
from usbmd.utils.config_validation import check_config, config_schema
from usbmd.utils.gui import App


class USBMDApp(App):
    """App class for building a GUI from a dictionary"""

    def __init__(self, ui=None, resolution=None, title=None, verbose=False):
        super().__init__(
            config_schema, resolution, title, button=False, verbose=verbose
        )
        self.ui = ui
        if self.ui:
            self.ui.gui = self

        self.user_info = set_data_paths()
        self.user_text = (
            f'User: {self.user_info["username"]} | '
            f'Hostname: {self.user_info["hostname"]}'
        )

        self.version_text = f"USBMD version: {usbmd.__version__}"
        self.data_root = f'data path: {self.user_info["data_root"]}'

        # Styling
        self.style.configure("ButtonFrame.TFrame", background=self.background_color)
        self.style.configure(
            "Footer.TLabel", font=("TkDefaultFont", 8), background=self.background_color
        )

        self.freeze_active_color = "#1193dd"
        self.style.map(
            "Freeze.TButton",
            foreground=[("active", "white"), ("disabled", "gray")],
            background=[("active", self.freeze_active_color), ("disabled", "#d9d9d9")],
            hovercolor=[("active", self.freeze_active_color), ("disabled", "#256d27")],
        )

        # Init
        self.version_label = None
        self.path_label = None
        self.user_label = None

        self.set_button = None
        self.run_button = None
        self.freeze_button = None
        self.load_button = None

        # pause functionality
        self.pause_var = False
        self.display_path_label = False

    def build(self, data):
        """Build the application"""
        self.version_label = ttk.Label(
            self, text=self.version_text, style="Footer.TLabel"
        )
        self.version_label.pack(side="bottom", padx=10, pady=0)

        if self.display_path_label:
            self.path_label = ttk.Label(
                self,
                text=self.data_root,
                font=("TkDefaultFont", 8),
                background=self.background_color,
            )
            self.path_label.pack(side="bottom", padx=10, pady=0)

        self.user_label = ttk.Label(self, text=self.user_text, style="Footer.TLabel")
        self.user_label.pack(side="bottom", padx=10, pady=5)

        self.build_right_panel()

        super().build(data)

    def build_right_panel(self):
        """Create the frame for the buttons"""
        button_frame = ttk.Frame(self, style="ButtonFrame.TFrame")

        # create the set button and pack it
        self.set_button = ttk.Button(button_frame, text="Set", command=self.set)
        self.set_button.pack(side="top", padx=5, pady=5)

        # create the run button and pack it
        self.run_button = ttk.Button(button_frame, text="Run", command=self.run)
        self.run_button.configure(style="Run.TButton")
        self.run_button.pack(side="top", padx=5, pady=5)

        # create the freeze button and pack it
        self.freeze_button = ttk.Button(
            button_frame, text="Freeze", command=self.freeze
        )
        self.freeze_button.configure(style="Freeze.TButton")
        self.freeze_button.pack(side="top", padx=5, pady=5)

        # create the load button and pack it
        self.load_button = ttk.Button(button_frame, text="Load", command=self.load)
        self.load_button.configure(style="Load.TButton")
        self.load_button.pack(side="top", padx=5, pady=5)

        # pack the button frame in the right panel
        button_frame.pack(side="right", fill="y", padx=5, pady=5)

        return button_frame

    def set(self):
        super().set()
        # pylint: disable=bad-option-value, unnecessary-dunder-call
        self.ui.__init__(self.data)

    def run(self):
        """TBA, run function"""
        print("Congratz, you clicked run.")
        # https://www.geeksforgeeks.org/how-to-use-thread-in-tkinter-python/
        if self.ui:
            # thread is necessary to not interfere with GUI mainloop
            try:
                self.style.configure("Run.TButton", background=self.button_color_active)
                thread = Thread(target=lambda: self.ui.run(plot=False))
                thread.start()
            except Exception as e:
                warnings.warn(f"Run failed: {e}")

            # can only plot after thread is finished and image is processed
            while thread.is_alive():
                pass

            self.style.configure("Run.TButton", background=self.button_color)
            self.ui.plot(self.ui.image, block=False, save=True, axis=True)

    def freeze(self):
        """Freeze function"""
        print("Congratz, you are now frozen.")
        if self.pause_var is False:
            self.pause_var = True
            self.style.configure("Freeze.TButton", background=self.freeze_active_color)
            print("pausing")
        else:
            self.pause_var = False
            self.style.configure("Freeze.TButton", background="#d9d9d9")
            print("resuming")

    def check_freeze(self):
        """Wait until pause var is set to 0"""
        while self.pause_var is True:
            pass

    def save_to_file(self, name=None):
        """Save config to file"""
        if name is None:
            try:
                name = self.ui.config.data.dataset_name
            except:
                name = None
        super().save_to_file(name)

    def load(self, data=None, entries=None):
        """Load in a new config file"""
        self.style.configure("Load.TButton", background=self.button_color_active)
        try:
            new_config = setup_config()
            super().load(new_config, self.entries)
        except:
            print("No new config loaded.")
        self.style.configure("Load.TButton", background=self.button_color)


if __name__ == "__main__":
    file = "./configs/config_picmus_rf.yaml"

    config = load_config_from_yaml(file)
    check_config(check_config(config))

    app = USBMDApp(title="USBMD GUI", resolution=(600, 300), verbose=True)
    app.build(config)
    app.mainloop()

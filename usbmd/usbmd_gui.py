"""USBMD GUI app, builds on the App class from usbmd/utils/gui.py

Provides a graphical user friendly way of interacting with the USBMD package
and setting config parameters.

TODO:
- add schema, such that options are available
- add option picker entry
- more error handling
- fix lists
- better tickboxes
- labels prettier
- fix load config, such that are inserted in entries

- **Author(s)**     : Tristan Stevens
- **Date**          : April 19th, 2023
"""

import asyncio
from tkinter import ttk
from typing import Dict, Optional, Tuple

import usbmd
from usbmd.common import set_data_paths
from usbmd.setup_usbmd import setup_config
from usbmd.utils import log
from usbmd.utils.config import load_config_from_yaml
from usbmd.utils.config_validation import check_config, config_schema
from usbmd.utils.gui import App


class USBMDApp(App):
    """App class for building a GUI from a dictionary"""

    def __init__(
        self,
        loop: Optional[object] = None,
        ui: Optional[object] = None,
        resolution: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        verbose: bool = False,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the USBMD GUI.

        Args:
            loop (object, optional): The event loop to use. Defaults to None.
            ui (object, optional): The user interface object. Defaults to None.
            resolution (tuple, optional): The resolution of the GUI window. Defaults to None.
            title (str, optional): The title of the GUI window. Defaults to None.
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
            config (dict, optional): The configuration settings. Defaults to None.
        """
        super().__init__(
            config_schema, resolution, title, button=False, verbose=verbose
        )

        if loop:
            self.loop = loop

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

        # run loop flag
        self.running = True

        if config:
            self.build(config)

    async def show(self):
        """Shows and updates the window (asyncronously)"""
        while self.running:
            self.update()
            await asyncio.sleep(0.1)

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
        self.run_button = ttk.Button(
            button_frame, text="Run", command=self._asyncio_task_wrapper(self.run)
        )
        self.run_button.configure(style="Run.TButton")
        self.run_button.pack(side="top", padx=5, pady=5)

        # create the freeze button and pack it
        self.freeze_button = ttk.Button(
            button_frame,
            text="Freeze",
            command=self._asyncio_task_wrapper(self.freeze),
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

    def _asyncio_task_wrapper(self, coro):
        """Wrap a coroutine in an asyncio task"""
        return lambda: self.loop.create_task(coro())

    def set(self):
        """Set config data and initialize the UI
        TODO: probably do not want to re-init the entire UI every time
        """
        super().set()
        # pylint: disable=bad-option-value, unnecessary-dunder-call
        self.ui.__init__(self.data)
        self.ui.gui = self

    async def run(self):
        """TBA, run function"""
        print("Congratz, you clicked run.")
        if self.ui:
            try:
                self.set()
                self.style.configure("Run.TButton", background=self.button_color_active)
                self.run_button.configure(state="disabled")

                self.ui.run(plot=True, block=False)
                self.run_button.configure(state="normal")
                self.style.configure("Run.TButton", background=self.button_color)

            except Exception as e:
                log.warning(f"Run failed: {e}")

    async def freeze(self):
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
        await asyncio.sleep(0.1)

    async def check_freeze(self):
        """Wait until pause var is set to 0"""
        while self.pause_var is True:
            await asyncio.sleep(0.1)

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
        log.warning(
            "Loading new config file, functionality not yet robustly implemented. "
            "Excpect some bugs..."
        )
        self.style.configure("Load.TButton", background=self.button_color_active)
        self.load_button.configure(state="disabled")
        new_config = setup_config()
        super().load(new_config, self.entries)
        self.style.configure("Load.TButton", background=self.button_color)
        self.load_button.configure(state="normal")

    def on_closing(self):
        """Close the app"""
        # destroy the app
        closed = super().on_closing()
        if closed is False:
            return
        # stop show update loop
        self.running = False
        # stop the asyncio loop
        self.loop.stop()


if __name__ == "__main__":
    file = "./configs/config_camus.yaml"

    config = load_config_from_yaml(file)
    check_config(check_config(config))

    app = USBMDApp(title="USBMD GUI", resolution=(600, 300), verbose=True)
    app.build(config)
    app.mainloop()

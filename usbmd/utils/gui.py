"""Easy GUI generation from config yaml.

- **Author(s)**     : Tristan Stevens
- **Date**          : April 27th, 2023
"""
import pprint
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from schema import Schema

from usbmd.utils.config import Config
from usbmd.utils.utils import filename_from_window_dialog, get_date_string


class App(tk.Tk):
    """App class for building a GUI from a dictionary"""

    def __init__(
        self,
        schema: Schema = None,
        resolution: str = None,
        title: str = None,
        button: bool = True,
        verbose: bool = False,
    ):
        super().__init__()

        self.schema = schema
        self.verbose = verbose
        self.debug = False

        self.data = None

        self.notebook = ttk.Notebook(self)
        self.entries = {}

        if title is not None:
            self.title(str(title))
        if resolution is not None:
            assert len(resolution) == 2
            self.geometry(f"{int(resolution[0])}x{int(resolution[1])}")
        else:
            self.geometry("600x200")

        # Set up style
        self.background_color = self.cget("background")
        self.notebook_color = "#e6e6e6"  # light gray
        self.button_color = "#d9d9d9"
        self.button_color_active = "#4baf4f"  # green

        self.styling()

        # Add save button
        if button:
            self.set_button = ttk.Button(self, text="Set", command=self.set)
            self.set_button.pack(side="right", padx=10, pady=10)
        else:
            self.set_button = None

        # these keys will trigger filedialog button (to select filepaths)
        self.keys_for_file_paths = [
            "filepath",
            "data_root",
            "file_path",
            "path",
            "datasets",
            "output",
            "repo_root",
        ]

        # on closing the window
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # bring to foreground
        self.raise_above_all()

    def build(self, data):
        """Build the application"""
        self.data = data
        if self.schema:
            self.data = self.schema.validate(Config(self.data).serialize())

        # Create tabs recursively using nested dictionaries in main dictionary
        self.add_tabs(self.data, self.entries, self.notebook)

        if self.set_button:
            self.set_button.pack()

        self.notebook.pack(expand=True, fill="both")

    def styling(self):
        """Initialize style"""
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TNotebook", background=self.background_color)
        self.style.configure("TNotebook.Tab", background="#b3b3b3")
        # self.style.configure("TLabel", background=self.notebook_color, foreground="#000000")
        self.style.configure("TEntry", background="#f2f2f2")
        self.style.map(
            "TButton",
            foreground=[("active", "white"), ("disabled", "gray")],
            background=[
                ("active", self.button_color_active),
                ("disabled", self.button_color),
            ],
            hovercolor=[("active", "#e6f2ff"), ("disabled", "#256d27")],
        )

        self.theme_bg = self.style.lookup(".", "background")

    def raise_above_all(self):
        """Move window to the foreground and center"""
        self.attributes("-topmost", 1)
        self.attributes("-topmost", 0)
        self.eval("tk::PlaceWindow . center")

    def add_tabs(self, dict_obj, entries, parent=None):
        """Build GUI with tabs for each nested dictionary"""
        if parent is None:
            parent = self.notebook

        for key, value in dict_obj.items():
            if key == "male":
                print("")
            if isinstance(value, dict):
                child_notebook = tk.ttk.Notebook(parent)
                child_notebook.pack(expand=True, fill="both")

                parent.add(child_notebook, text=key)
                entries[key] = {}
                self.add_tabs(value, entries[key], child_notebook)
            else:
                frame = ttk.Frame(parent)
                label = ttk.Label(frame, text=key + ":")

                entry = self.create_entry(frame, key, value)

                # Pack label and entry widgets
                label.pack(side="left", padx=10, pady=10)
                entry.pack(side="left", padx=10, pady=10)

                parent.add(frame, text=key)
                entries[key] = entry

    def create_entry(self, frame, key, value):
        """Create entry widget based on default value type"""
        if key in self.keys_for_file_paths:
            entry = ttk.Entry(frame)
            entry.configure(validatecommand=(entry.register(self.validate_int), "%P"))
            entry.type = Path
            # Create an open file button
            open_button = ttk.Button(
                frame, text="Select file", command=lambda: self.insert_file_path(entry)
            )
            open_button.pack(side="right", padx=10, pady=10)
        elif isinstance(value, str):
            entry = ttk.Entry(frame)
            entry.type = str
        elif isinstance(value, bool):
            entry = BooleanEntry(frame, value)  # <-- create this class
        elif isinstance(value, int):
            entry = ttk.Entry(frame, validate="key")
            entry.configure(validatecommand=(entry.register(self.validate_int), "%P"))
            entry.type = int
        elif isinstance(value, float):
            entry = ttk.Entry(frame, validate="key")
            entry.configure(validatecommand=(entry.register(self.validate_float), "%P"))
            entry.type = float
        elif value is None:
            entry = ttk.Entry(frame, validate="key")
            entry.type = lambda x: x
        elif isinstance(value, list):
            entry = ListEntry(frame, value)
        else:
            raise ValueError(
                f"Unsupported data type {type(value)} for default value {key} : {value}"
            )

        if not isinstance(value, (list, bool)):
            entry.insert(0, str(value))
        return entry

    @staticmethod
    def validate_int(value):
        """Validate if entry is an integer"""
        try:
            if value:
                int(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_float(value):
        """Validate if entry is a float"""
        try:
            if value:
                float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_path(value):
        """Validate if entry is a valid path"""
        try:
            if value:
                Path(value)
            return True
        except ValueError:
            return False

    def set(self):
        """When set button is pressed update dictionary"""
        self.save(self.data, self.entries)

    def save(self, data, entries):
        """Update the dictionary with the new values"""

        def _save(data, entries):
            for key, entry in entries.items():
                if isinstance(entry, dict):
                    _save(data[key], entries[key])
                else:
                    old_val = data[key]
                    new_val = self._get_entry_val(entry, key)
                    if old_val != new_val:
                        try:
                            data[key] = new_val
                            if self.schema:
                                self.schema.validate(dict(self.data))
                            if self.verbose:
                                print(f"Updating {key} from {old_val} to {new_val}")
                        except Exception as e:
                            data[key] = old_val
                            print(f"Couldn't update {key} to {new_val}: {e}")

        _save(data, entries)

        if self.debug:
            pprint.pprint(data)
            print("-----------------")

    def load(self, data, entries):
        """Load the dictionary into the GUI"""

        def _load(data, entries):
            for key, value in data.items():
                if isinstance(value, dict):
                    _load(data[key], entries[key])
                else:
                    entries[key].delete(0, tk.END)
                    entries[key].insert(1, str(data[key]))

        _load(data, entries)
        self.data = data
        self.save(self.data, entries)

    @staticmethod
    def _get_entry_val(entry, key):
        value = entry.get()

        if value == "None":
            return None
        try:
            value = entry.type(value)
        except Exception as e:
            print(f"Error converting {key} with value {value} to type {entry.type}")
            print(e)
        return value

    @staticmethod
    def insert_file_path(entry):
        """Insert the filepath from filedialog in a textfield"""
        try:
            filename = filename_from_window_dialog()
        except:
            return
        entry.delete(0, tk.END)
        entry.insert(1, filename)

    def on_closing(self):
        """Pop up closing dialog when closing window"""
        if tk.messagebox.askokcancel("Quit", "Do you want to quit?"):
            if tk.messagebox.askyesno("Save", "Save config?"):
                self.save_to_file()
            self.quit()
            self.destroy()

    def save_to_file(self, name=None):
        """Save current data to file"""
        if name is None:
            name = "_"
        else:
            name = "_" + name + "_"

        folder = Path("./custom_configs")
        folder.mkdir(exist_ok=True)
        filename = folder / (get_date_string() + name + "config.yaml")
        Config(self.data).save_to_yaml(filename)
        print(f"Succesfully saved config to {filename}")


class BooleanEntry(tk.Frame):
    """Frame class for a Boolean entry."""

    def __init__(self, frame, value):
        super().__init__(frame)
        self.type = bool
        self.var = tk.BooleanVar(value=value)
        self.checkbox = ttk.Checkbutton(self, variable=self.var)
        self.checkbox.pack(side="left", padx=10, pady=10)

    def get(self):
        """Get the value of the entry"""
        return self.var.get()

    def delete(self, first, last=None):
        """Delete the entry"""
        raise NotImplementedError

    def insert(self, first, last=None):
        """Delete the entry"""
        raise NotImplementedError


class ListEntry(ttk.Frame):
    """Frame class for a List entry."""

    def __init__(self, frame, value):
        super().__init__(frame)
        self.type = list
        self.entries = []

        for i, item in enumerate(value):
            if isinstance(item, str):
                entry = ttk.Entry(self)
                entry.type = str
            elif isinstance(item, bool):
                entry = ttk.Combobox(self, values=["True", "False"])
                entry.type = bool
            elif isinstance(item, int):
                entry = ttk.Entry(self, validate="key")
                entry.configure(
                    validatecommand=(entry.register(self.validate_int), "%P")
                )
                entry.type = int
            elif isinstance(item, float):
                entry = ttk.Entry(self, validate="key")
                entry.configure(
                    validatecommand=(entry.register(self.validate_float), "%P")
                )
                entry.type = float
            else:
                raise ValueError(f"Unsupported data type for list item: {type(item)}")

            entry.insert(0, str(item))
            entry.grid(row=i, column=0, padx=10, pady=10)
            self.entries.append(entry)

        add_button = ttk.Button(self, text="+", command=self.add_entry)
        add_button.grid(row=len(value), column=0, padx=10, pady=10)

        remove_button = ttk.Button(self, text="-", command=self.remove_entry)
        remove_button.grid(row=len(value), column=1, padx=10, pady=10)

    def add_entry(self):
        """Add a new entry to the list"""
        entry = ttk.Entry(self)
        entry.type = str
        entry.grid(row=len(self.entries), column=0, padx=10, pady=10)
        self.entries.append(entry)

    def remove_entry(self):
        """Remove the last entry from the list"""
        if self.entries:
            entry = self.entries.pop()
            entry.destroy()

    @staticmethod
    def validate_int(value):
        """Validate if entry is an integer"""
        try:
            if value:
                int(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def validate_float(value):
        """Validate if entry is a float"""
        try:
            if value:
                float(value)
            return True
        except ValueError:
            return False

    def get(self):
        """Return the list of values"""
        return [entry.type(entry.get()) for entry in self.entries]

    def delete(self, first, last=None):
        """Delete the entry"""
        raise NotImplementedError

    def insert(self, first, last=None):
        """Insert an entry"""
        raise NotImplementedError


if __name__ == "__main__":
    data = {
        "general": {"title": "My App", "version": 1.0, "description": "A simple app"},
        "user": {
            "name": "John",
            "age": 30.2,
            "you": [1.0, 2, 3, 4],
            "filepath": "somepath/var",
            "address": {
                "street": "123 Main St",
                "city": "New York",
                "state": "NY",
                "zip": "10001",
                "male": True,
            },
        },
    }

    app = App()
    app.build(data)
    app.mainloop()

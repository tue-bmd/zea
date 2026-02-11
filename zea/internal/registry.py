"""Registration module for registering classes and their names to be able to
refer to them by name in config files. The module contains a decorator class
for registering classes that can be used to register a name and optionally
additional values to keys for the class.

Usage:
- In the file defining a base class and possibly the subclasses, import the
RegisterDecorator class and create a decorator object. In the items_to_register
argument, pass a list of strings that will be used as keys for the additional
keys to register values for for every registered item.
- For each subclass, decorate the class with the decorator object and pass the
name to register to the class as the first argument and optionally additional
values to register to the keys for the class as keyword arguments.
- In other code that needs to use these classes import only the registry object
and use the registry object to get the class corresponding to a name.

Example:
```datasets.py
dataset_registry = RegisterDecorator(items_to_register=['probe_used', 'scan_class'])

@dataset_registry(name='picmus', probe_used='L11-5V', scan_class=PicmusScan)
class PICMUS(Dataset):
    ...
```

In another file:
```other_file.py
from zea.data import dataset_registry

dataset_class = dataset_registry['picmus']
dataset = dataset_class()
```
"""


class RegisterDecorator:
    """Decorator class for registering classes.

    The docorator registers a name to the class and optionally registers
    additional values to keys for the class.
    """

    def __init__(self, items_to_register=None):
        # The registry is a dictionary mapping names to classes
        self.registry = {}

        # Register additional values to keys for the class
        # additional_registries is a dictionary mapping registry names to
        # dictionaries mapping classes to values (yeah that's a mouthful)
        self.additional_registries = {}

        if items_to_register is None:
            items_to_register = {}

        for reg in items_to_register:
            assert isinstance(reg, str), "Item to register must be a string"
            self.additional_registries[reg.lower()] = {}

    def __call__(self, name, **kwargs):
        """The decorator function.

        The name is the name to register to the class and the kwargs are the
        additional values to register to the class.

        Note: All names and keys are converted to lowercase.
        """
        assert isinstance(name, str), "Name must be a string"
        assert name not in self.registry, f"Name {name} already registered"

        call_kwargs = kwargs.copy()
        name = name.lower()

        def _register(cls, name=name):
            self.registry[name] = cls

            for regname, value in call_kwargs.items():
                # If there is an additional registry with name regname,
                # register the value to the class.
                if regname in self.additional_registries:
                    # Add the class as key In the additional registry with
                    # name regname and the value as value
                    self.additional_registries[regname][cls] = value

            return cls

        return _register

    def get_parameter(self, cls_or_name, parameter):
        """Get parameter.

        Returns the value of the parameter for the class with the given
        class or name. This value can be a string or a class type.
        """
        if isinstance(cls_or_name, str):
            cls_or_name = self.registry[cls_or_name.lower()]
        # Assert that key is a class type
        assert isinstance(cls_or_name, type) or callable(cls_or_name), (
            "Key must be a class type or function"
        )
        return self.additional_registries[parameter.lower()][cls_or_name]

    def __str__(self) -> str:
        """String representation of the registry.

        Prints the keys and class names of the registry each on a single
        line followed by the keys and values of each additional registry.
        """
        string = "registry:\n"
        for key, cls in self.registry.items():
            string += f"{key.ljust(30)}: {cls.__name__}\n"

        string += "\nadditional_registries:\n"
        for reg, dictionary in self.additional_registries.items():
            string += f"{reg}:\n"
            for cls, val in dictionary.items():
                string += f"\t{cls.__name__.ljust(30)}: {val}\n"

        return string

    def __getitem__(self, key):
        """Returns the class corresponding to the key.

        The key can be a string or a class type.
        """
        assert isinstance(key, str), "Key must be a string"
        try:
            return self.registry[key.lower()]
        except KeyError as exc:
            raise KeyError(
                f"Name {key} not registered. Please choose from {self.registered_names()}."
            ) from exc

    def get_name(self, cls):
        """Retrieve the registry key name associated with the given class or instance.
        Subclasses of the class are also considered."""
        # If class is an instance, get its type
        if not isinstance(cls, type):
            cls = type(cls)

        # First check if the class is directly
        for name, registered_class in self.registry.items():
            if cls is registered_class:
                return name

        # If not found, check if the class is a subclass of any registered class
        for name, registered_class in self.registry.items():
            if issubclass(cls, registered_class):
                return name

        raise KeyError(f"Class {cls} not registered.")

    def get_additional_registries(self):
        """Returns a list of the names of the additional registries."""
        return list(self.additional_registries.keys())

    def registered_names(self):
        """Returns a list of the names registered."""
        return list(self.registry.keys())

    def __contains__(self, key):
        """Returns True if the key is registered."""
        return key.lower() in self.registry

    def __iter__(self):
        """Returns an iterator over the keys of the registry."""
        return iter(self.registry)

    def clear(self):
        """Clears the registry."""
        self.registry = {}
        self.additional_registries = {}
        items_to_register = self.additional_registries.keys()

        if items_to_register is None:
            items_to_register = {}

        for reg in items_to_register:
            self.additional_registries[reg.lower()] = {}

    def filter_by_argument(self, argument, value):
        """Filter the registry by the given argument and value.

        Returns a list of names of classes that have the given value for the
        given argument.
        """
        return [
            name
            for name, cls in self.registry.items()
            if self.get_parameter(cls, argument) == value
        ]


# The registry for the probes.
probe_registry = RegisterDecorator()

# The registry for the beamformers.
# separate registries to avoid dupicate entries
# as torch and tf beamformers cannot coexist in same registry
tf_beamformer_registry = RegisterDecorator(items_to_register=["name", "framework"])

torch_beamformer_registry = RegisterDecorator(items_to_register=["name", "framework"])

metrics_registry = RegisterDecorator(items_to_register=["name", "paired", "jittable"])

checks_registry = RegisterDecorator(items_to_register=["data_type"])
ops_registry = RegisterDecorator(items_to_register=["name"])
ops_dep_registry = RegisterDecorator(items_to_register=["name"])
model_registry = RegisterDecorator(items_to_register=["name"])
diffusion_guidance_registry = RegisterDecorator(items_to_register=["name"])
operator_registry = RegisterDecorator(items_to_register=["name"])
action_selection_registry = RegisterDecorator(items_to_register=["name"])

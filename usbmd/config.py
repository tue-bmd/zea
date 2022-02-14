import yaml
from pathlib import Path
import copy

class Config(dict):
    """Config class.
    
    This Config class extends a normal dictionary with the getattr and 
    setattr functionality, and it enables saving to a yml.
    
    """
    
    def __init__(self, dictionary):
        super().__init__(dictionary)

        for key, value in self.items():
            assert(key not in ['keys', 'values', 'items']), 'The configuration contains the following key {key} which is reserved already as a standard attribute of a dict.'
            
            # Change the key: TODO: now only string keys are supported.
            # new_key = str(key).replace('-', '_')
            #self.update({new_key, self.pop(key)})
            
            
            if isinstance(value, list) or isinstance(value, tuple):
                detected_dict = 0
                for idx, val in enumerate(value):
                    if isinstance(val, dict):
                        val = Config(val)
                        self[key][idx] = val
                        #setattr(self, key, val)
                        detected_dict += 1
                if not detected_dict:
                    setattr(self, key, value)

            elif isinstance(value, dict):
                value = Config(value)
                setattr(self, key, value)
            else:
                setattr(self, key, value)
            

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def set_nested_key(self, config_entry, value):
        """[summary]

        Args:
            config_entry (str): String of the format <key1>.<key2>.<key3> etc. A maximum depth of 4 keys is currently supported.
            value (tuple, list, int, float): A value to set for this nested key. 

        Raises:
            NotImplementedError: [description]
        """
        
        nested_keys = config_entry.split(".")
        if len(nested_keys) == 1:
            self[nested_keys[0]] = value
        elif len(nested_keys) == 2:
            self[nested_keys[0]][nested_keys[1]] = value
        elif len(nested_keys) == 3:
            self[nested_keys[0]][nested_keys[1]][nested_keys[2]]  = value
        elif len(nested_keys) == 4:
            self[nested_keys[0]][nested_keys[1]][nested_keys[2]][nested_keys[3]] = value
        else:
            raise NotImplementedError

    def serialize(self):
        dictionary = {}
        for key, value in self.items():
            if isinstance(value, Config):
                dictionary[key] = value.serialize()
            # elif isinstance(value, list) or isinstance(value,tuple):
            #     for idx,
            else:
                dictionary[key] = value
        return dictionary

    def deep_copy(self):
        return Config(copy.deepcopy(self.serialize()))

    def save_to_yaml(self, path):
        with open(Path(path), 'w') as save_file:
            yaml.dump(self.serialize(), save_file, default_flow_style=False)

def load_config_from_yaml(path):
    dictionary = yaml.load(open(Path(path)), Loader=yaml.FullLoader)
    if dictionary:
        return Config(dictionary)
    else:
        return {}
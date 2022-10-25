import os
import tqdm
import numpy as np
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from PIL import Image

def filename_from_window_dialog(window_name=None, filetypes=None, initialdir=None):
    """ Get filename through dialog window
    Args:
        window_name: string with name of window
        filetypes: tuple of tuples containing (name, filetypes)
            example: 
                (('mat or h5 or whatever you want', '*.mat *.hdf5 *'), (ckpt, *.ckpt))
        initialdir: path to directory where window will start
    Returns:
        filename: string containing path to selected file
        
    """
    if filetypes is None:
        filetypes = (('all files', '*.*'),)
        
    root = Tk()
    # open in foreground
    root.wm_attributes('-topmost', 1)
    # we don't want a full GUI, so keep the root window from appearing
    root.withdraw()
    # show an "Open" dialog box and return the path to the selected file
    filename = askopenfilename(
        parent=root, 
        title=window_name, 
        filetypes=filetypes, 
        initialdir=initialdir,
    )
    # check whether a file was selected
    if filename:
        return Path(filename)
    else:
        raise Exception('No file selected.')

def translate(array, range_from, range_to):
    """ Map values in array from one range to other. 
    
    Args:
        array (ndarray): input array.
        range_from (Tuple): lower and upper bound of original array.
        range_to (Tuple): lower and upper bound to which array should be mapped.
        
    Returns:
        (ndarray): translated array
    """
    leftMin, leftMax = range_from
    rightMin, rightMax = range_to
    if leftMin == leftMax: 
        return np.ones_like(array) * rightMax
    
    # Convert the left range into a 0-1 range (float)
    valueScaled = (array - leftMin) / (leftMax - leftMin)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * (rightMax - rightMin))

def search_file_tree(directory, filetypes=None, write=True):
    """Lists all files in directory and sub-directories.

    If file_paths.txt is detected in the directory, that file is read and used.
    
    Args:
        directory (str): path to directory.
        filetypes (Tuple of strings, optional): filetypes. 
            Defaults to image types (.png etc.).
        write (bool, optional): Whether to write to file. Useful has searching
            the tree takes quite a while. Defaults to True.

    Returns:
        file_paths (List): List with str to all file paths.

    """
    directory = Path(directory)
    if (directory / 'file_paths.txt').is_file():
        print('Using pregenerated txt file in the following directory for reading file paths: ')
        print(directory)
        with open(directory / 'file_paths.txt') as file:
            file_paths = file.read().splitlines() 
        return file_paths
    
    # set default file type
    if filetypes is None:
        filetypes = ('jpg', 'jpeg', 'JPEG', 'png', 'PNG')
    
    file_paths = []
    
    # Traverse file tree to index all dicom files
    for dirpath, _, filenames in tqdm.tqdm(os.walk(directory), desc='Searching file tree'):
        for file in filenames:
            # Append to file_paths if it is a filetype file
            if file.endswith(filetypes):
                file_paths.append(str(Path(dirpath)/Path(file)))
    
    print(f'\nFound {len(file_paths)} image files in .\\{Path(directory)}\n')
    assert len(file_paths) > 0, 'ERROR: No image files were found'
    
    if write:
        # np.savetxt(directory / 'file_paths.txt', file_paths, delimiter='\n', fmt='%s')
        with open(directory / 'file_paths.txt', 'w') as file:
            file_paths = [file + '\n' for file in file_paths]
            file.writelines(file_paths)
            
    return file_paths

def find_key(dictionary, contains, case_sensitive=False):
    """Find key in dictionary that contains partly the string `contains`

    Args:
        dictionary (dict): Dictionary to find key in.
        contains (str): String which the key should .
        case_sensitive (bool, optional): Whether the search is case sensitive. 
            Defaults to False.

    Returns:
        str: the key of the dictionary that contains the query string.

    """
    if case_sensitive:
        key = [k for k in dictionary.keys() if contains in k]
    else:
        key = [k for k in dictionary.keys() if contains in k.lower()]
    return key[0]
    
def plt_window_has_been_closed(ax):
    """Checks whether matplotlib plot window is closed"""
    fig = ax.figure.canvas.manager
    active_fig_managers = plt._pylab_helpers.Gcf.figs.values()
    return fig not in active_fig_managers

def print_clear_line():
    """Clears line. Helpful when printing in a loop on the same line."""
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    print(LINE_UP, end=LINE_CLEAR)

def to_image(image, range: tuple=None, pillow: bool=True):
    """Convert numpy array to uint8 image format.

    Args:
        image (ndarray): input array image
        range (tuple, optional): assumed range of input data. 
            Defaults to None.
        pillow (bool, optional): whether to convert the image 
            array to pillow object. Defaults to True.

    Returns:
        image: output image array uint8 [0, 255] 
            (pillow if set to True)
    """
    if range:
        image = translate(
            np.clip(image, *range), range, (0, 255)
        )
    
    image = image.astype(np.uint8)
    if pillow:
        image = Image.fromarray(image)
    return image
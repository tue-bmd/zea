# Convert image dataset to hdf5 in USBMD format

# algo:
# - run os.walk on the dir
# - any image file, take the (current root - overall root) and make a new dir path, 
#     take all the images, save each image as a h5.
# - optionally, take a group_by regex to group paths into frames.

import os 

from PIL import Image
import numpy as np

from usbmd.data_format.usbmd_data_format import generate_usbmd_dataset

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']

def img_to_h5(existing_dataset_root, new_dataset_root, current_dir, files, dataset_name):
    # Make a new directory in new_dataset_root to match the directory tree in existing_dataset_root
    relative_dir = os.path.relpath(current_dir, existing_dataset_root)
    new_dir_path = f'{new_dataset_root}/{relative_dir}/'
    if not os.path.exists(new_dir_path): os.makedirs(new_dir_path)

    # Select only image files
    img_files = filter(lambda path: any(path.lower().endswith(ext) for ext in IMG_EXTENSIONS), files)


    # If there's a grouping key, make a list of lists of files
    # For each list if there's a sorting key, sort the files
    # For each list, read all images and stack them on axis=0 to make n_frames.


    for img_file in img_files:
        # Read image
        img_path = f'{current_dir}/{img_file}'
        image = np.array(Image.open(img_path).convert('L')) # Read grayscale = 1 channel
        image = image[None, ...] # add n_frames dimension

        # Create new h5 file
        img_file_without_extension, _ = os.path.splitext(img_file)
        new_h5_file_path = f'{new_dir_path}/{img_file_without_extension}.hdf5'
        generate_usbmd_dataset(
            path=new_h5_file_path,
            image=image,
            probe_name="generic",
            description=f"{dataset_name or 'image'} dataset converted to USBMD format",
        )

def convert_image_dataset(existing_dataset_root, new_dataset_root, dataset_name=None):
    for curent_dir, _, files in os.walk(existing_dataset_root):
        img_to_h5(existing_dataset_root, new_dataset_root, curent_dir, files, dataset_name)

if __name__ == "__main__":
    convert_image_dataset('/mnt/z/Ultrasound-BMd/data/oisin/camus_test', '/home/oinolan/latent-ultrasound-diffusion/camus_test_h5')
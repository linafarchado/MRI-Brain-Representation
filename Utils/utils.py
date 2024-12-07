import torch
import torch.nn.functional as F
import os
import nibabel as nib
import numpy as np

def filter_black_images_from_dataset(dataset):
    filtered_dataset = []
    for item in dataset:
        if isinstance(item, tuple):
            image, label = item
            if torch.sum(image) != 0 and torch.sum(label) != 0:
                filtered_dataset.append((image, label))
        else:
            image = item
            if torch.sum(image) != 0:
                filtered_dataset.append(image)
    return filtered_dataset

def pad_image(image, target_height=160, target_width=192):
    if len(image.shape) == 2:
        height, width = image.shape
    elif len(image.shape) == 3:
        _, height, width = image.shape
    else:
        raise ValueError("Check the dimensions: Image must be 2D or 3D")

    # Calculate padding sizes
    pad_height = target_height - height
    pad_width = target_width - width

    # Apply padding (top, bottom, left, right)
    padding = (0, pad_width, 0, pad_height)  # Only pad bottom and right
    padded_image = F.pad(image, padding, mode='constant', value=0)
    
    return padded_image

def save_image_nifti(image, filename, outputs):
    os.makedirs(outputs, exist_ok=True)
    image_nifti = nib.Nifti1Image(image, np.eye(4))
    image_path_img = os.path.join(outputs, f'{filename}-T2.img')
    nib.save(image_nifti, image_path_img)

def save_label_nifti(label, filename, outputs):
    os.makedirs(outputs, exist_ok=True)
    label_nifti = nib.Nifti1Image(label, np.eye(4))
    label_path_img = os.path.join(outputs, f'{filename}-label.img')
    nib.save(label_nifti, label_path_img)

def add_latent_noise(latent, noise_range=(-0.01, 0.01)):
    noise_vector = torch.zeros_like(latent).uniform_(noise_range[0], noise_range[1])
    noisy_latent = latent + noise_vector
    return noisy_latent

def create_binary_mask(image, lower_bound=0.4, upper_bound=0.5):
    mask = (image >= lower_bound) & (image <= upper_bound)
    return mask.float()
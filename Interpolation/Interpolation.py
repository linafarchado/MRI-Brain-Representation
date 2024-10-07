from ConvAutoencoder import ConvAutoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import nibabel as nib
import numpy as np
import os
import random
import sys
import torch.nn.functional as F

# Ajoutez le chemin du dossier parent au sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Utils import filter_black_images_from_dataset
from Visualize import visualize_interpolation, visualize_interpolation_even
from Dataloader import CustomDataset

class ImageInterpolator:
    def __init__(self, model):
        self.model = model
    
    def interpolate(self, image1, image2, alpha=random.uniform(0, 1)):
        # Encode the images
        latent1 = self.model.encode(image1)
        latent2 = self.model.encode(image2)
        
        # Interpolation
        interpolated_latent = alpha * latent1 + (1 - alpha) * latent2

        # Decode the interpolated latent
        interpolated_image = self.model.decode(interpolated_latent)
        
        return interpolated_image

def save_image_nifti(image, filename, outputs):
    os.makedirs(outputs, exist_ok=True)
    image_nifti = nib.Nifti1Image(image, np.eye(4))
    image_path_img = os.path.join(outputs, f'{filename}_T2.img')
    nib.save(image_nifti, image_path_img)
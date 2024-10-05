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

def main(folder, outputs, checkpoint):
    model = ConvAutoencoder(out_channels=32)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    image_interpolator = ImageInterpolator(model)

    # Load the validation dataset
    validation_data_folder = folder
    val_dataset = CustomDataset(validation_data_folder)
    val_dataset = filter_black_images_from_dataset(val_dataset)
    random.shuffle(val_dataset)

    cpt = 0
    # Interpolate between two images
    for i in range(len(val_dataset) - 1):
        image1 = val_dataset[i]
        image2 = val_dataset[i + 1]
        alpha = 0.5

        # Ensure the images are 4D tensors (batch, channel, height, width)
        if image1.dim() != 4:
            image1 = image1.unsqueeze(0)
        if image2.dim() != 4:
            image2 = image2.unsqueeze(0)
        
        # If the images don't have a channel dimension, add it
        if image1.size(1) != 1:
            image1 = image1.unsqueeze(1)
        if image2.size(1) != 1:
            image2 = image2.unsqueeze(1)

        # Apply interpolation and visualize the results
        interpolated_image = image_interpolator.interpolate(image1, image2, alpha)
        interpolated_image = interpolated_image.squeeze(0)
        interpolated_image_np = interpolated_image.permute(1, 2, 0).detach().cpu().numpy()

        #save_image_nifti(interpolated_image_np, f'interpolated_{i}', outputs)
        cpt += 1
        visualize_interpolation(cpt, image1, image2, interpolated_image, model_name='ConvAutoencoder')


def main_even_images(folder, checkpoint, outputs):
    model = ConvAutoencoder(out_channels=32)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    image_interpolator = ImageInterpolator(model)

    # Load the validation dataset
    validation_data_folder = folder
    val_dataset = CustomDataset(validation_data_folder)

    # Évaluer l'interpolation
    avg_mse = evaluate_interpolation(model, val_dataset, image_interpolator, outputs)
    print(f"Average MSE for interpolation: {avg_mse}")
    
if __name__ == "__main__":
    #main(folder="../Training", outputs="TestImages", checkpoint="CNN32.pth")
    main_even_images(folder="../Training", checkpoint="CNN32.pth", outputs="InterpolatedEvenImages")
import os
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

import sys
# Ajoutez le chemin du dossier parent au sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from Utils import pad_image

class CustomDataset(Dataset):
    def __init__(self, folder):
        self.black_images_folder = "Black_images"
        os.makedirs(self.black_images_folder, exist_ok=True)
        self.folder = folder
        self.files_list = sorted([filename for filename in os.listdir(folder) if 'T2' in filename])
        self.loaded_images = []
        self.lst_patient_idx = []
        self.load_images()
        self.normalize()

    def load_images(self):
        for idx_patient, filename in enumerate(self.files_list):
            img_path = os.path.join(self.folder, filename)
            img = nib.load(img_path).get_fdata()
            if img.shape[2] == 1:
                # Si la troisième dimension est 1, on squeeze directement
                tensor_image = torch.tensor(img.squeeze())
                if torch.any(tensor_image != 0):
                    self.loaded_images.append(torch.tensor(tensor_image).unsqueeze(0))
                else:
                    self.save_black_image(img.squeeze(), idx_patient)
            else:
                for it_image in range(img.shape[0]):
                    tensor_image = torch.tensor(img[:, :, it_image])
                    if torch.any(tensor_image != 0):
                        self.loaded_images.append(torch.tensor(img[:, :, it_image]).unsqueeze(0))
                    else:
                        self.save_black_image(np.expand_dims(img[:, :, it_image], axis=2), idx_patient)

            self.lst_patient_idx.append(idx_patient)

    def save_black_image(self, image, idx_patient):
        # Sauvegarde de l'image noire dans le dossier "Black_images"
        image_name = f"black_image_patient{idx_patient}_slice{len(self.loaded_images)}.png"
        image_path = os.path.join(self.black_images_folder, image_name)
        plt.imsave(image_path, image, cmap='gray')

    def normalize(self):
        self.lst_patient_mu = []
        self.lst_patient_sigma = []
        for image in self.loaded_images:
            non_zero_mask = image != 0
            mean = torch.mean(image[non_zero_mask])
            std = torch.std(image[non_zero_mask])
            self.lst_patient_mu.append(mean)
            self.lst_patient_sigma.append(std)
            if std != 0:
                normalized_image = (image[non_zero_mask] - mean) / (std * 5)
                normalized_image = normalized_image / 2 + 0.5
                image[non_zero_mask] = torch.clip(normalized_image, 0, 1)

    def __len__(self):
        return len(self.loaded_images)

    def __getitem__(self, idx):
        image = self.loaded_images[idx].squeeze().float()
        # Pad image to target size
        image = pad_image(image)
        return image
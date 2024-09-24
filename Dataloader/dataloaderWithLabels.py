import os
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

import sys

# Ajoutez le chemin du dossier parent au sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Utils import pad_image

class CustomDatasetWithLabels(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files_list = sorted([filename for filename in os.listdir(folder) if 'T2' in filename])
        self.labels_list = sorted([filename for filename in os.listdir(folder) if 'label' in filename])
        self.loaded_images = []
        self.loaded_labels = []
        self.lst_patient_idx = []
        self.load_images()
        self.load_labels()
        self.normalize()
    
    def load_images(self):
        for idx_patient, filename in enumerate(self.files_list):
            img_path = os.path.join(self.folder, filename)
            img = nib.load(img_path).get_fdata()
            for it_image in range(img.shape[0]):
                self.loaded_images.append(torch.tensor(img[:, :, it_image]).unsqueeze(0))
                self.lst_patient_idx.append(idx_patient)
    
    def load_labels(self):
        for filename in self.labels_list:
            label_path = os.path.join(self.folder, filename)
            label = nib.load(label_path).get_fdata()
            for it_image in range(label.shape[0]):
                self.loaded_labels.append(torch.tensor(label[:, :, it_image]).unsqueeze(0).long())
    
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
        label = self.loaded_labels[idx].squeeze().long()

        # Pad image to target size
        image = pad_image(image)
        label = pad_image(label)

        label[label == 10] = 1
        label[label == 150] = 2
        label[label == 250] = 3
        return image, label

    def get_num_classes(self):
        all_labels = torch.cat(self.loaded_labels)
        return len(torch.unique(all_labels))
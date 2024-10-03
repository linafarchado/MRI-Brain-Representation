import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import sys

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Utils import pad_image

class CustomDatasetWithLabelsFiltered(Dataset):
    def __init__(self, folder, is_training=True):
        self.folder = folder
        self.is_training = is_training
        self.files_list, self.labels_list = self._filter_files()
        self.loaded_images = []
        self.loaded_labels = []
        self.image_patient_mapping = []
        self.load_images()
        self.load_labels()
        self.normalize()
    
    def _filter_files(self):

        all_images = sorted([filename for filename in os.listdir(self.folder) if 'T2' in filename and filename.endswith('.hdr')])
        all_labels = sorted([filename for filename in os.listdir(self.folder) if 'label' in filename and filename.endswith('.hdr')])
        
        filtered_images = []
        filtered_labels = []
        
        for img_file, label_file in zip(all_images, all_labels):
            # Extract patient number from filename
            patient_num = int(img_file.split('-')[1])
            
            # Training: patients 1-8, Testing: patients 9-10
            if self.is_training and 1 <= patient_num <= 8:
                filtered_images.append(img_file)
                filtered_labels.append(label_file)
            elif not self.is_training and 9 <= patient_num <= 10:
                filtered_images.append(img_file)
                filtered_labels.append(label_file)
        
        return filtered_images, filtered_labels
    
    def load_images(self):
        for idx_patient, filename in enumerate(self.files_list):
            img_path = os.path.join(self.folder, filename)
            img = nib.load(img_path).get_fdata()
            if img.shape[2] == 1:
                tensor_image = torch.tensor(img.squeeze()).unsqueeze(0)
                if torch.sum(tensor_image) != 0:
                    self.loaded_images.append(tensor_image)
                    self.image_patient_mapping.append((idx_patient, len(self.loaded_images) - 1))
            else:
                for it_image in range(img.shape[0]):
                    tensor_image = torch.tensor(img[:, :, it_image]).unsqueeze(0)
                    if torch.sum(tensor_image) != 0:
                        self.loaded_images.append(tensor_image)
                        self.image_patient_mapping.append((idx_patient, len(self.loaded_images) - 1))

    def load_labels(self):
        for filename in self.labels_list:
            label_path = os.path.join(self.folder, filename)
            label = nib.load(label_path).get_fdata()
            if label.shape[2] == 1:
                tensor_label = torch.tensor(label.squeeze()).unsqueeze(0).long()
                if torch.sum(tensor_label) != 0:
                    self.loaded_labels.append(tensor_label)
            else:
                for it_image in range(label.shape[0]):
                    tensor_label = torch.tensor(label[:, :, it_image]).unsqueeze(0).long()
                    if torch.sum(tensor_label) != 0:
                        self.loaded_labels.append(tensor_label)
        
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

        # Remap label values
        label[label == 10] = 1
        label[label == 150] = 2
        label[label == 250] = 3
        
        return image, label

    def get_num_classes(self):
        all_labels = torch.cat(self.loaded_labels)
        return len(torch.unique(all_labels))
    
    def get_patient_idx(self, idx):
        for patient_idx, image_idx in self.image_patient_mapping:
            if image_idx == idx:
                return patient_idx
        return None
import torch
import torch.nn.functional as F
import os
import nibabel as nib
import numpy as np
import pandas as pd

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
    print(f"Image shape: {image.shape}")
    print(f"Target shape, h, w: ({height}, {width})")
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

def process_seg_data(data, group):
    # Create lists to store results
    classes = []
    weights = []
    min_values = []
    max_values = []
    
    # Process each class
    for classe, weights_dict in data[group].items():
        # Exclude 'Base' from weights
        weight_data = {k: v for k, v in weights_dict.items() if k != 'Base'}
        
        # Find weights with min and max values
        min_weight = min(weight_data, key=lambda x: weight_data[x]['min'])
        max_min_weight = max(weight_data, key=lambda x: weight_data[x]['min'])
        min_max_weight = min(weight_data, key=lambda x: weight_data[x]['max'])
        max_max_weight = max(weight_data, key=lambda x: weight_data[x]['max'])
        
        # Append results
        classes.extend([classe]*4)
        weights.extend([
            f'Smallest Min Weight: {min_weight}', 
            f'Largest Min Weight: {max_min_weight}', 
            f'Smallest Max Weight: {min_max_weight}', 
            f'Largest Max Weight: {max_max_weight}'
        ])
        min_values.extend([
            weight_data[min_weight]['min'],
            weight_data[max_min_weight]['min'],
            weight_data[min_max_weight]['min'],
            weight_data[max_max_weight]['min']
        ])
        max_values.extend([
            weight_data[min_weight]['max'],
            weight_data[max_min_weight]['max'],
            weight_data[min_max_weight]['max'],
            weight_data[max_max_weight]['max']
        ])

    # Create DataFrame
    df = pd.DataFrame({
        'Classe': classes,
        'Weight Description': weights,
        'Min Value': min_values,
        'Max Value': max_values
    })
    
    return df

def process_seg_accuracy_data(data, group):
    # Exclude 'Base' from weights
    weights_dict = {k: v for k, v in data[group].items() if k != 'Base'}
    
    # Find weights with min and max values
    min_weight = min(weights_dict, key=lambda x: weights_dict[x]['min'])
    max_min_weight = max(weights_dict, key=lambda x: weights_dict[x]['min'])
    min_max_weight = min(weights_dict, key=lambda x: weights_dict[x]['max'])
    max_max_weight = max(weights_dict, key=lambda x: weights_dict[x]['max'])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Weight Description': [
            f'Smallest Min Weight: {min_weight}', 
            f'Largest Min Weight: {max_min_weight}', 
            f'Smallest Max Weight: {min_max_weight}', 
            f'Largest Max Weight: {max_max_weight}'
        ],
        'Min Value': [
            weights_dict[min_weight]['min'],
            weights_dict[max_min_weight]['min'],
            weights_dict[min_max_weight]['min'],
            weights_dict[max_max_weight]['min']
        ],
        'Max Value': [
            weights_dict[min_weight]['max'],
            weights_dict[max_min_weight]['max'],
            weights_dict[min_max_weight]['max'],
            weights_dict[max_max_weight]['max']
        ]
    })
    
    return df


def process_base_values(data, group):
    # Locate the Base entry
    base_value = data[group].get('Base', {})
    
    # Create DataFrame with Base values
    df = pd.DataFrame({
        'Base Min': [base_value.get('min')],
        'Base Max': [base_value.get('max')]
    })
    
    return df

def process_base_values_dice(data, group):
    # Lists to store results
    classes = []
    min_values = []
    max_values = []
    
    # Process each class
    for classe, weights_dict in data[group].items():
        # Extract base values
        base_values = weights_dict.get('Base', {})
        
        # Append results
        classes.append(classe)
        min_values.append(base_values.get('min'))
        max_values.append(base_values.get('max'))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Classe': classes,
        'Base Min': min_values,
        'Base Max': max_values
    })
    
    return df
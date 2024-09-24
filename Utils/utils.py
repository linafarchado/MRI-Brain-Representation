import torch
import torch.nn.functional as F

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


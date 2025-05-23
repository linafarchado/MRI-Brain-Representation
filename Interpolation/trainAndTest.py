import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# Ajoutez le chemin du dossier parent au sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Utils import save_image_nifti, add_latent_noise
from Visualize import visualize_interpolation_even, visualize_interpolation, visualize_multi_interpolation, visualize_two_images_interpolation

# test of even images range(2, len(dataset) - 2, 2) or odd range(1, len(dataset) - 1, 2)
def test(load, dataset, model, device, noise_range=(-0.01, 0.01), noise=False):
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), f'{load}.pth')))
    model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0
    count = 0

    for i in tqdm(range(1, len(dataset) - 1, 2)):
        if dataset.get_patient_idx(i) == dataset.get_patient_idx(i+2):
            img1 = dataset[i].to(device)
            img2 = dataset[i+1].to(device)
            img3 = dataset[i+2].to(device)

            latent1 = model.encode(img1)
            latent2 = model.encode(img2)
            latent3 = model.encode(img3)
            
            alpha = round(random.uniform(0, 1), 2)

            interpolated_latent = (1 - alpha) * latent1 + alpha * latent3

            # Add noise to interpolated latent
            noisy_interpolated_latent = add_latent_noise(interpolated_latent, noise_range) if noise else interpolated_latent
            
            interpolated_image = model.decode(noisy_interpolated_latent)

            mse = mse_loss(interpolated_image, img2)
            total_mse += mse.item()
            count += 1

            if interpolated_image.dim() == 4:
                interpolated_image = interpolated_image.squeeze(0)
            interpolated_image_np = interpolated_image.permute(1, 2, 0).detach().cpu().numpy()

            #save_image_nifti(interpolated_image_np, f'interpolated_{i}', f'{load}ImagesEVENnoise')

            # Visualisation
            #visualize_interpolation_even(i+1, dataset, alpha, interpolated_image, model_name='ConvAutoencoder', save_dir=f'{load}PlotEVENnoise')
            visualize_two_images_interpolation(i+1, dataset, alpha, interpolated_image, model_name='ConvAutoencoder', save_dir=f'{load}PlotODDV2')

    avg_mse = total_mse / count if count > 0 else 0
    return avg_mse

def process_one_batch(model, img1, img2, img3, optimizer, device, noise_range=(-0.01, 0.01), noise=False):
    model.train()
    if optimizer is not None:
        optimizer.zero_grad()
    
    latent1 = model.encode(img1)
    latent2 = model.encode(img2)
    latent3 = model.encode(img3)
    
    # Make sure there is always 0.5, 0.25 and 0.75 in the alphas
    alphas = torch.cat([
        torch.rand(1),
        torch.tensor([0.5]),
        torch.tensor([0.25, 0.75]),
    ]).to(device)
    
    mse_loss = nn.MSELoss()
    total_loss = 0
    for alpha in alphas:
        interpolated_latent = (1 - alpha) * latent1 + alpha * latent3
        
        # Add noise to interpolated latent
        noisy_interpolated_latent = add_latent_noise(interpolated_latent, noise_range) if noise else interpolated_latent
        
        generated_img = model.decode(noisy_interpolated_latent)
        
        if alpha == 0.5:
            target_img = img2
        else:
            target_img = (1 - alpha) * img1 + alpha * img3
        
        if generated_img.dim() == 4:
            generated_img = generated_img.squeeze(0)

        if generated_img.dim() == 3:
            generated_img = generated_img.squeeze(0)

        loss = mse_loss(generated_img, target_img)

        total_loss += loss
    
    avg_loss = total_loss / len(alphas)
    
    if optimizer is not None:
        avg_loss.backward()
        optimizer.step()
    
    return avg_loss.item()

def train(model, train_dataset, optimizer, device, noise_range=(-0.01, 0.01), noise=False):
    model.train()
    total_loss = 0
    batch_count = 0
    
    for i in tqdm(range(2, len(train_dataset) - 2, 2)):
        if train_dataset.get_patient_idx(i) == train_dataset.get_patient_idx(i+2):
            img1 = train_dataset[i].to(device)
            img2 = train_dataset[i+1].to(device)
            img3 = train_dataset[i+2].to(device)
            
            avg_batch_loss = process_one_batch(model, img1, img2, img3, optimizer, device, noise_range=noise_range, noise=noise)
            
            total_loss += avg_batch_loss
            batch_count += 1
            
    return total_loss / batch_count if batch_count > 0 else float('inf')

def evaluate(model, val_dataset, device, noise_range=(-0.01, 0.01), noise=False):
    model.eval()
    total_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for i in range(2, len(val_dataset) - 2, 2):
            if val_dataset.get_patient_idx(i) == val_dataset.get_patient_idx(i+2):
                img1 = val_dataset[i].to(device)
                img2 = val_dataset[i+1].to(device)
                img3 = val_dataset[i+2].to(device)
                
                avg_batch_loss = process_one_batch(model, img1, img2, img3, optimizer=None, device=device, noise_range=noise_range, noise=noise)
                total_loss += avg_batch_loss
                batch_count += 1
    
    return total_loss / batch_count if batch_count > 0 else float('inf')

def test_random_images(load, dataset, model, device):
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), f'{load}.pth')))
    model.eval()

    length = len(dataset)
    for i in range(length):
        j = random.randint(0, length - 1)
        img1 = dataset[i].to(device)
        img2 = dataset[j].to(device)

        latent1 = model.encode(img1)
        latent2 = model.encode(img2)
        
        alpha = round(random.uniform(0, 1), 2)

        interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
        
        interpolated_image = model.decode(interpolated_latent)


        if interpolated_image.dim() == 4:
            interpolated_image = interpolated_image.squeeze(0)
        #interpolated_image_np = interpolated_image.permute(1, 2, 0).detach().cpu().numpy()

        #save_image_nifti(interpolated_image_np, f'interpolated_{i}', f'{load}Images')

        # Visualisation
        visualize_interpolation(i, j, img1, img2, alpha, interpolated_image, save_dir='InterpolatedImagesPlot')

# test of odd images
"""def test(load, dataset, model, device, noise_range=(-0.01, 0.01), noise=False):
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), f'{load}.pth')))
    model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0
    count = 0

    for i in tqdm(range(1, len(dataset) - 1, 2)):
        if dataset.get_patient_idx(i) == dataset.get_patient_idx(i+2):
            img1 = dataset[i].to(device)
            img2 = dataset[i+1].to(device)
            img3 = dataset[i+2].to(device)

            latent1 = model.encode(img1)
            latent2 = model.encode(img2)
            latent3 = model.encode(img3)
            
            alpha = round(random.uniform(0, 1), 2)

            interpolated_latent = (1 - alpha) * latent1 + alpha * latent3

            # Add noise to interpolated latent
            noisy_interpolated_latent = add_latent_noise(interpolated_latent, noise_range) if noise else interpolated_latent
            
            interpolated_image = model.decode(noisy_interpolated_latent)

            mse = mse_loss(interpolated_image, img2)
            total_mse += mse.item()
            count += 1

            if interpolated_image.dim() == 4:
                interpolated_image = interpolated_image.squeeze(0)
            interpolated_image_np = interpolated_image.permute(1, 2, 0).detach().cpu().numpy()

            save_image_nifti(interpolated_image_np, f'interpolated_{i}', f'{load}ImagesBis')

            # Visualisation
            visualize_interpolation_even(i+1, dataset, alpha, interpolated_image, model_name='ConvAutoencoder', save_dir=f'{load}PlotBis')

    avg_mse = total_mse / count if count > 0 else 0
    return avg_mse"""

# test of 4 images, 2 from each side of the image
def generate_weighted_random_alphas(num_points):
    """
    Generate random alpha values that sum to 1, with higher weights for central points
    """
    mid_point = num_points // 2
    weights = [1 - abs(i - mid_point) / (num_points + 1) for i in range(num_points)]

    # Generate random values and apply weights
    random_values = [random.random() * weight for weight in weights]
    
    # Normalize to sum to 1
    total = sum(random_values)
    alphas = [val / total for val in random_values]
    
    return [round(alpha, 2) for alpha in alphas]

# test of 4 images, 2 from each side of the image
"""def test(load, dataset, model, device):
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), f'{load}.pth')))
    model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0
    count = 0

    for i in tqdm(range(2, len(dataset) - 2)):
        if (dataset.get_patient_idx(i-2) == dataset.get_patient_idx(i-1) == 
            dataset.get_patient_idx(i) == dataset.get_patient_idx(i+1) == 
            dataset.get_patient_idx(i+2)):
            
            # Get 5 consecutive images
            img0 = dataset[i-2].to(device)
            img1 = dataset[i-1].to(device)
            img2 = dataset[i].to(device)
            img3 = dataset[i+1].to(device)
            img4 = dataset[i+2].to(device)

            # Encode all images
            latent0 = model.encode(img0)
            latent1 = model.encode(img1)
            latent2 = model.encode(img2)
            latent3 = model.encode(img3)
            latent4 = model.encode(img4)

            alphas = generate_weighted_random_alphas(4)

            interpolated_latent = (
                alphas[0] * latent0 + 
                alphas[1] * latent1 + 
                alphas[2] * latent3 + 
                alphas[3] * latent4
            )
            
            interpolated_image = model.decode(interpolated_latent)

            mse = mse_loss(interpolated_image, img2)
            total_mse += mse.item()
            count += 1

            if interpolated_image.dim() == 4:
                interpolated_image = interpolated_image.squeeze(0)
            interpolated_image_np = interpolated_image.permute(1, 2, 0).detach().cpu().numpy()

            save_image_nifti(interpolated_image_np, f'interpolated_{i}', f'{load}ImagesMulti')
            
            visualize_multi_interpolation(
                i, 
                dataset, 
                alphas, 
                interpolated_image, 
                model_name='ConvAutoencoder', 
                save_dir=f'{load}PlotMulti'
            )

    avg_mse = total_mse / count if count > 0 else 0
    return avg_mse"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import os


import sys

# Ajoutez le chemin du dossier parent au sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Interpolation import save_image_nifti
from Visualize import visualize_interpolation_even

def test(load, dataset, model, device):
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), f'{load}.pth')))
    model.eval()
    mse_loss = nn.MSELoss()
    total_mse = 0
    count = 0

    for i in tqdm(range(2, len(dataset) - 2, 2)):
        if dataset.get_patient_idx(i) == dataset.get_patient_idx(i+2):
            img1 = dataset[i].to(device)
            img2 = dataset[i+1].to(device)
            img3 = dataset[i+2].to(device)

            # Encoder les images avec le modèle entraîné
            latent1 = model.encode(img1)
            latent2 = model.encode(img2)
            latent3 = model.encode(img3)
            
            alpha = round(random.uniform(0, 1), 2)

            # Interpolation dans l'espace latent
            interpolated_latent = (1 - alpha) * latent1 + alpha * latent3
            
            # Décodage pour obtenir l'image interpolée
            interpolated_image = model.decode(interpolated_latent)

            print(f'Shape of interpolated image: {interpolated_image.shape}')
            print(f'Shape of image: {dataset[i+1].shape}')
            mse = mse_loss(interpolated_image, img2)
            total_mse += mse.item()
            count += 1

            if interpolated_image.dim() == 4:
                interpolated_image = interpolated_image.squeeze(0)
            interpolated_image_np = interpolated_image.permute(1, 2, 0).detach().cpu().numpy()

            save_image_nifti(interpolated_image_np, f'interpolated_{i}', f'{load}Images')

            # Visualisation
            visualize_interpolation_even(i+1, dataset, mse, alpha, interpolated_image, model_name='ConvAutoencoder', save_dir=f'{load}Plot')

    avg_mse = total_mse / count if count > 0 else 0
    return avg_mse

def process_one_batch(model, img1, img2, img3, optimizer, device):
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
        generated_img = model.decode(interpolated_latent)
        
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

def train(model, train_dataset, optimizer, device):
    model.train()
    total_loss = 0
    batch_count = 0
    
    for i in tqdm(range(2, len(train_dataset) - 2, 2)):
        if train_dataset.get_patient_idx(i) == train_dataset.get_patient_idx(i+2):
            img1 = train_dataset[i].to(device)
            img2 = train_dataset[i+1].to(device)
            img3 = train_dataset[i+2].to(device)
            
            avg_batch_loss = process_one_batch(model, img1, img2, img3, optimizer, device)
            
            total_loss += avg_batch_loss
            batch_count += 1
            
    return total_loss / batch_count if batch_count > 0 else float('inf')

def evaluate(model, val_dataset, device):
    model.eval()
    total_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for i in range(2, len(val_dataset) - 2, 2):
            if val_dataset.get_patient_idx(i) == val_dataset.get_patient_idx(i+2):
                img1 = val_dataset[i].to(device)
                img2 = val_dataset[i+1].to(device)
                img3 = val_dataset[i+2].to(device)
                
                avg_batch_loss = process_one_batch(model, img1, img2, img3, optimizer=None, device=device)
                total_loss += avg_batch_loss
                batch_count += 1
    
    return total_loss / batch_count if batch_count > 0 else float('inf')
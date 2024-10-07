import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio

def visualize_interpolation_models(cpt, image1, image2, interpolated_image8, interpolated_image16, interpolated_image32):
    # Convert tensors to numpy arrays for visualization
    image1_np = image1.detach().cpu().numpy().squeeze()
    image2_np = image2.detach().cpu().numpy().squeeze()
    interpolated_image8_np = interpolated_image8.detach().cpu().numpy().squeeze()
    interpolated_image16_np = interpolated_image16.detach().cpu().numpy().squeeze()
    interpolated_image32_np = interpolated_image32.detach().cpu().numpy().squeeze()

    save_dir='InterpolatedImagesPlot'
    os.makedirs(save_dir, exist_ok=True)

    # Plot the images
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    axes[0, 0].axis('off')
    axes[0, 1].imshow(image1_np, cmap='gray')
    axes[0, 1].set_title('Image 1')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(image2_np, cmap='gray')
    axes[0, 2].set_title('Image 2')
    axes[0, 2].axis('off')

    # Plot interpolated images in the bottom row
    axes[1, 0].imshow(interpolated_image8_np, cmap='gray')
    axes[1, 0].set_title('Interpolated Image (8 channels)')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(interpolated_image16_np, cmap='gray')
    axes[1, 1].set_title('Interpolated Image (16 channels)')
    axes[1, 1].axis('off')
    axes[1, 2].imshow(interpolated_image32_np, cmap='gray')
    axes[1, 2].set_title('Interpolated Image (32 channels)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    
    # Save the plot to a file
    save_path = os.path.join(save_dir, f'interpolated_images{cpt}.png')
    plt.savefig(save_path)
    
    plt.show()

# visualize one model
def visualize_interpolation(cpt, image1, image2, interpolated_image, model_name, save_dir='InterpolatedImagesPlot'):
    # Convert tensors to numpy arrays for visualization
    image1_np = image1.detach().cpu().numpy().squeeze()
    image2_np = image2.detach().cpu().numpy().squeeze()
    interpolated_image_np = interpolated_image.detach().cpu().numpy().squeeze()

    os.makedirs(save_dir, exist_ok=True)

    # Plot the images
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].axis('off')
    axes[1].imshow(image1_np, cmap='gray')
    axes[1].set_title('Image 1')
    axes[1].axis('off')
    axes[2].imshow(image2_np, cmap='gray')
    axes[2].set_title('Image 2')
    axes[2].axis('off')

    # Plot interpolated images in the bottom row
    axes[0].imshow(interpolated_image_np, cmap='gray')
    axes[0].set_title(f'Interpolated Image ({model_name})')
    axes[0].axis('off')

    plt.tight_layout()
    
    # Save the plot to a file
    save_path = os.path.join(save_dir, f'interpolated_images{cpt}.png')
    plt.savefig(save_path)
    
    plt.show()

def visualize_segmentation(image, true_label, pred_label, idx, folder):
    os.makedirs(folder, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(image[0], cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(true_label, cmap='viridis')
    ax2.set_title('True Segmentation')
    ax2.axis('off')
    
    ax3.imshow(pred_label, cmap='viridis')
    ax3.set_title('Predicted Segmentation')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{idx}.png"))
    plt.show()
    plt.close()

def visualize_image(dataset, idx, folder):
    os.makedirs(folder, exist_ok=True)
    image, label = dataset[idx]
    
    plt.figure(figsize=(10, 5))
    
    # Image plot
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Image IRM')
    plt.axis('off')

    # Label plot
    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap='gray')
    plt.title('Label Correspondant')
    plt.axis('off')
    plt.savefig(os.path.join(folder, f"image_{idx+1}.png"))
    plt.show()

def visualize_image_without_label(image, idx, folder):
    os.makedirs(folder, exist_ok=True)
    plt.figure(figsize=(5, 5))
    
    # Image plot
    plt.imshow(image, cmap='gray')
    plt.title('Image IRM')
    plt.axis('off')
    plt.savefig(os.path.join(folder, f"image_{idx+1}.png"))
    plt.show()


def visualize_interpolation_even(center_idx, dataset, alpha, interpolated_image, model_name, save_dir='InterpolatedEvenImagesPlot'):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mse_loss = torch.nn.MSELoss()
    psnr = PeakSignalNoiseRatio().to(device)

    # Déterminer les indices à visualiser en restant dans les limites du dataset
    dataset_length = len(dataset)
    indices = [center_idx-2, center_idx-1, center_idx, center_idx+1, center_idx+2]
    indices = [idx for idx in indices if 0 <= idx < dataset_length]

    # Récupérer les images et calculer les métriques
    images = []
    mse_values = []
    psnr_values = []
    diff_images = []

    for idx in indices:
        img = dataset[idx].unsqueeze(0)
        images.append(img)
        mse_values.append(mse_loss(interpolated_image.to(device), img.to(device)).item())
        psnr_values.append(psnr(interpolated_image.to(device), img.to(device)).item())
        diff_images.append(torch.abs(interpolated_image.to(device) - img.to(device)).squeeze().detach().cpu().numpy())

    # Convert tensors to numpy arrays for visualization
    images_np = [img.squeeze().detach().cpu().numpy() for img in images]
    interpolated_image_np = interpolated_image.squeeze().detach().cpu().numpy()

    # Plot the images
    fig, axes = plt.subplots(3, 3, figsize=(15, 18))

    # Top row: Images i-2, i, i+2 (si disponibles)
    top_indices = [0, 2, 4]
    for i, idx in enumerate(top_indices):
        if idx < len(images_np):
            axes[0, i].imshow(images_np[idx], cmap='gray')
            axes[0, i].set_title(f'Image {indices[idx]}, MSE: {mse_values[idx]:.4f}')
        axes[0, i].axis('off')

    # Bottom row: Difference images (si disponibles)
    for i, idx in enumerate(top_indices):
        if idx < len(diff_images):
            axes[1, i].imshow(diff_images[idx], cmap='viridis')
            axes[1, i].set_title(f'Difference Image {indices[idx]}, PSNR: {psnr_values[idx]:.2f}')
        axes[1, i].axis('off')

    # Middle row: Image i-1, Interpolated, i+1 (si disponibles)
    middle_indices = [1, -1, 3]  # -1 pour interpolated image
    for i, idx in enumerate(middle_indices):
        if idx == -1:
            axes[2, i].imshow(interpolated_image_np, cmap='gray')
            axes[2, i].set_title(f'Interpolated Image ({model_name})')
        elif idx < len(images_np):
            axes[2, i].imshow(images_np[idx], cmap='gray')
            axes[2, i].set_title(f'Image {indices[idx]}, alpha: {alpha if i == 0 else 1 - round(alpha, 2)}')
        axes[2, i].axis('off')

    plt.tight_layout()

    # Save the plot to a file
    save_path = os.path.join(save_dir, f'interpolated_images_{center_idx}.png')
    plt.savefig(save_path)

    plt.close()

def visualize_segmentation_Dice(image, true_label, pred_label, idx, dice_pred, patient_idx, folder):
    os.makedirs(folder, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Affichage de l'image originale
    ax1.imshow(image[0], cmap='gray')
    ax1.set_title(f'Original Image from patient {patient_idx}')
    ax1.axis('off')
    
    # Affichage de la vérité de terrain avec le score Dice
    ax2.imshow(true_label, cmap='viridis')
    ax2.set_title('True Segmentation')
    ax2.axis('off')
    
    # Ajouter le score Dice sur la vérité de terrain
    dice_text = '\n'.join([f'Class {cls}: {score:.2f}' for cls, score in dice_pred])
    ax2.text(0.5, 0.5, f'Dice Scores:\n{dice_text}', color='white', fontsize=10, ha='center', va='center', 
             bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    
    # Affichage de la segmentation prédite
    ax3.imshow(pred_label, cmap='viridis')
    ax3.set_title('Predicted Segmentation')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"{idx}.png"))
    plt.show()
    plt.close()

def plot_dice_vs_std(dice_scores, std_values, save_path='dice_vs_std.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(std_values, dice_scores, marker='o', color='b', label='Dice Score')
    plt.axvline(x=0, color='r', linestyle='--', label='Ecart-type = 0')

    plt.title('Dice moyen en fonction de l\'écart-type')
    plt.xlabel('Écart-type')
    plt.ylabel('Dice moyen')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()

def visualize_interpolation_even_with_labels(center_idx, dataset, alpha, interpolated_image, model_name, label=True, labelFile=False, save_dir='InterpolatedEvenImagesPlot'):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mse_loss = torch.nn.MSELoss()
    psnr = PeakSignalNoiseRatio().to(device)

    # Déterminer les indices à visualiser en restant dans les limites du dataset
    dataset_length = len(dataset)
    indices = [center_idx-2, center_idx-1, center_idx, center_idx+1, center_idx+2]
    indices = [idx for idx in indices if 0 <= idx < dataset_length]

    # Récupérer les images et calculer les métriques
    images = []
    mse_values = []
    psnr_values = []
    diff_images = []

    for idx in indices:
        img = dataset[idx][1].unsqueeze(0) if label else dataset[idx][0].unsqueeze(0)
        images.append(img)
        mse_values.append(mse_loss(interpolated_image.to(device), img.to(device)).item())
        psnr_values.append(psnr(interpolated_image.to(device), img.to(device)).item())
        diff_images.append(torch.abs(interpolated_image.to(device) - img.to(device)).squeeze().detach().cpu().numpy())

    # Convert tensors to numpy arrays for visualization
    images_np = [img.squeeze().detach().cpu().numpy() for img in images]
    interpolated_image_np = interpolated_image.squeeze().detach().cpu().numpy()

    # Plot the images
    fig, axes = plt.subplots(3, 3, figsize=(15, 18))

    # Top row: Images i-2, i, i+2 (si disponibles)
    top_indices = [0, 2, 4]
    for i, idx in enumerate(top_indices):
        if idx < len(images_np):
            axes[0, i].imshow(images_np[idx], cmap='gray')
            axes[0, i].set_title(f'Image {indices[idx]}, MSE: {mse_values[idx]:.4f}')
        axes[0, i].axis('off')

    # Bottom row: Difference images (si disponibles)
    for i, idx in enumerate(top_indices):
        if idx < len(diff_images):
            axes[1, i].imshow(diff_images[idx], cmap='viridis')
            axes[1, i].set_title(f'Difference Image {indices[idx]}, PSNR: {psnr_values[idx]:.2f}')
        axes[1, i].axis('off')

    # Middle row: Image i-1, Interpolated, i+1 (si disponibles)
    middle_indices = [1, -1, 3]  # -1 pour interpolated image
    for i, idx in enumerate(middle_indices):
        if idx == -1:
            axes[2, i].imshow(interpolated_image_np, cmap='gray')
            axes[2, i].set_title(f'Interpolated Image ({model_name})')
        elif idx < len(images_np):
            axes[2, i].imshow(images_np[idx], cmap='gray')
            axes[2, i].set_title(f'Image {indices[idx]}, alpha: {alpha if i == 0 else 1 - alpha}')
        axes[2, i].axis('off')

    plt.tight_layout()

    # Save the plot to a file
    filename = f'interpolated_label_{center_idx}' if labelFile else f'interpolated_images_{center_idx}'
    save_path = os.path.join(save_dir, f'{filename}.png')
    plt.savefig(save_path)

    plt.close()
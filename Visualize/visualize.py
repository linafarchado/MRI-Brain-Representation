import os
import matplotlib.pyplot as plt
import numpy as np

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
def visualize_interpolation(cpt, image1, image2, interpolated_image, model_name):
    # Convert tensors to numpy arrays for visualization
    image1_np = image1.detach().cpu().numpy().squeeze()
    image2_np = image2.detach().cpu().numpy().squeeze()
    interpolated_image_np = interpolated_image.detach().cpu().numpy().squeeze()

    save_dir='InterpolatedImagesPlot'
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

def visualize_image_without_label(image, folder, idx):
    os.makedirs(folder, exist_ok=True)
    plt.figure(figsize=(5, 5))
    
    # Image plot
    plt.imshow(image, cmap='gray')
    plt.title('Image IRM')
    plt.axis('off')
    plt.savefig(os.path.join(folder, f"image_{idx+1}.png"))
    plt.show()
import matplotlib.pyplot as plt
import os

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
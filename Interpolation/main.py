import sys
import os
import matplotlib.pyplot as plt
# Ajoutez le chemin du dossier parent au sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Utils import create_binary_mask
from Dataloader import CustomDataset, CustomDatasetWithLabelsFiltered

if __name__ == "__main__":
    
    folder_training = "../Training/"
    folder_interpolated = "../NEWINTERPOLATIONSAVE"

    # Chargement des datasets
    dataset_training = CustomDatasetWithLabelsFiltered(folder_training, is_training=True)
    interpolated_dataset = CustomDatasetWithLabelsFiltered(folder_interpolated, interpolation=True)
    """print('Nombre d\'images dans les images d\'entraînement:', len(dataset_training))
    print('Nombre d\'images dans les images interpolées:', len(interpolated_dataset))
    sum = len(dataset_training) + len(interpolated_dataset)
    print('Nombre total d\'images:', sum)
    print("training: ", 0.8 * sum)
    print("validation: ", 0.2 * sum)"""
    

    # Récupération des pixels pour chaque dataset
    training_pixels = dataset_training.get_all_pixels()
    interpolated_pixels = interpolated_dataset.get_all_pixels()
    print('Nombre de pixels dans les images d\'entraînement:', len(training_pixels))
    print('Nombre de pixels dans les images interpolées:', len(interpolated_pixels))

    folder = "maskedInterpolatedImages"
    os.makedirs(folder, exist_ok=True)

    for idx in range(5):
        #interpolated_image, interpolated_label = interpolated_dataset[idx]
        interpolated_image, interpolated_label = interpolated_dataset[idx]
        
        mask = create_binary_mask(interpolated_image, lower_bound=0.4, upper_bound=0.5)
        
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        fig.patch.set_facecolor('lightgray')
        
        axes[0].imshow(interpolated_image.numpy(), cmap='gray')
        axes[0].set_title("Image interpolée")
        axes[0].axis('off')
        
        axes[1].imshow(mask.numpy(), cmap='gray')
        axes[1].set_title("Masque binaire (0.4 - 0.5)")
        axes[1].axis('off')

        axes[2].imshow(interpolated_label, cmap='viridis')
        axes[2].set_title("Label interpolé")
        axes[2].axis('off')
        
        plt.tight_layout()
        output_path = f"{folder}/masked_{idx}.png"
        plt.savefig(output_path)

    """
        

    # Création des histogrammes
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(training_pixels, bins=256, color='blue', alpha=0.7, log=True)
    plt.title("Histogramme des pixels - Images d'entraînement")
    plt.xlabel("Valeur du pixel")
    plt.ylabel("Fréquence (log)")
    plt.xlim(0, 0.5)

    plt.subplot(1, 2, 2)
    plt.hist(interpolated_pixels, bins=256, color='green', alpha=0.7, log=True)
    plt.title("Histogramme des pixels - Images interpolées")
    plt.xlabel("Valeur du pixel")
    plt.ylabel("Fréquence (log)")
    plt.xlim(0, 0.5)
    
    # Sauvegarde de la figure
    plt.tight_layout()
    output_path = "histogrammes_pixels_log_NEWINTERPOLATION.png"
    plt.savefig(output_path)
    print(f"Les histogrammes ont été sauvegardés dans le fichier {output_path}")
    # Visualize pixel values
    # Création des histogrammes
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(training_pixels, bins=256, color='blue', alpha=0.7)
    plt.title("Histogramme des pixels - Images d'entraînement")
    plt.xlabel("Valeur du pixel")
    plt.ylabel("Fréquence")

    plt.subplot(1, 2, 2)
    plt.hist(interpolated_pixels, bins=256, color='green', alpha=0.7)
    plt.title("Histogramme des pixels - Images interpolées")
    plt.xlabel("Valeur du pixel")
    plt.ylabel("Fréquence")


    # Sauvegarde de la figure au lieu de l'afficher
    plt.tight_layout()
    output_path = "histogrammes_pixels_NEWINTERPOLATION.png"  # Chemin du fichier de sortie
    plt.savefig(output_path)
    print(f"Les histogrammes ont été sauvegardés dans le fichier {output_path}")"""
    """
    print(len(dataset))
    for i in range(120, 145):
        img = dataset[i]
        print(f"shape: {img.shape}")
        print(f"Visualisation de l'image {i+1}")
        visualize_image_without_label(dataset[i],"mages", i)
    """
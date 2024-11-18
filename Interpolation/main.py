import sys
import os
import matplotlib.pyplot as plt
# Ajoutez le chemin du dossier parent au sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Visualize import visualize_image_without_label
from Utils import filter_black_images_from_dataset
from Dataloader import CustomDataset, CustomDatasetWithLabelsFiltered

if __name__ == "__main__":
    folder_training = "../Training/"
    folder_interpolatedODD = "../InterpolationSavedLabelsODD/"

    # Chargement des datasets
    dataset_training = CustomDatasetWithLabelsFiltered(folder_training)
    interpolated_datasetODD = CustomDatasetWithLabelsFiltered(folder_interpolatedODD, interpolation=True)
    print('Nombre d\'images dans les images d\'entraînement:', len(dataset_training))
    print('Nombre d\'images dans les images interpolées:', len(interpolated_datasetODD))

    # Récupération des pixels pour chaque dataset
    training_pixels = dataset_training.get_all_pixels()
    interpolated_pixelsODD = interpolated_datasetODD.get_all_pixels()
    print('Nombre de pixels dans les images d\'entraînement:', len(training_pixels))
    print('Nombre de pixels dans les images interpolées:', len(interpolated_pixelsODD))

    # Création des histogrammes
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(training_pixels, bins=256, color='blue', alpha=0.7, log=True)
    plt.title("Histogramme des pixels - Images d'entraînement")
    plt.xlabel("Valeur du pixel")
    plt.ylabel("Fréquence (log)")
    plt.xlim(0, 0.5)

    plt.subplot(1, 2, 2)
    plt.hist(interpolated_pixelsODD, bins=256, color='green', alpha=0.7, log=True)
    plt.title("Histogramme des pixels - Images interpolées")
    plt.xlabel("Valeur du pixel")
    plt.ylabel("Fréquence (log)")
    plt.xlim(0, 0.5)
    
    # Sauvegarde de la figure
    plt.tight_layout()
    output_path = "histogrammes_pixels_log.png"
    plt.savefig(output_path)
    print(f"Les histogrammes ont été sauvegardés dans le fichier {output_path}")

    # Visualize pixel values    
    """

    # Création des histogrammes
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(training_pixels, bins=256, color='blue', alpha=0.7)
    plt.title("Histogramme des pixels - Images d'entraînement")
    plt.xlabel("Valeur du pixel")
    plt.ylabel("Fréquence")

    plt.subplot(1, 2, 2)
    plt.hist(interpolated_pixelsODD, bins=256, color='green', alpha=0.7)
    plt.title("Histogramme des pixels - Images interpolées")
    plt.xlabel("Valeur du pixel")
    plt.ylabel("Fréquence")


    # Sauvegarde de la figure au lieu de l'afficher
    plt.tight_layout()
    output_path = "histogrammes_pixels.png"  # Chemin du fichier de sortie
    plt.savefig(output_path)
    print(f"Les histogrammes ont été sauvegardés dans le fichier {output_path}")
    """
    """
    print(len(dataset))
    for i in range(120, 145):
        img = dataset[i]
        print(f"shape: {img.shape}")
        print(f"Visualisation de l'image {i+1}")
        visualize_image_without_label(dataset[i],"mages", i)
    """
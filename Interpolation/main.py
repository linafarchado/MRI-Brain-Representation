import sys
import os

# Ajoutez le chemin du dossier parent au sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Visualize import visualize_image_without_label
from Utils import filter_black_images_from_dataset
from Dataloader import CustomDataset, CustomDatasetWithLabelsFiltered

if __name__ == "__main__":
    folder_training = "../Training/"
    folder_interpolated = "../InterpolationSavedLabels/"

    dataset_training = CustomDatasetWithLabelsFiltered(folder_training)
    print(f"Taille du dataset d'entraînement: {len(dataset_training)}")
    dataset_testing = CustomDatasetWithLabelsFiltered(folder_training, is_training=False)
    print(f"Taille du dataset de test: {len(dataset_testing)}")
    

    interpolated_dataset = CustomDatasetWithLabelsFiltered(folder_interpolated, interpolation=True)
    print(f"Taille du dataset interpolé: {len(interpolated_dataset)}")

    """
    print(len(dataset))
    for i in range(120, 145):
        img = dataset[i]
        print(f"shape: {img.shape}")
        print(f"Visualisation de l'image {i+1}")
        visualize_image_without_label(dataset[i],"mages", i)
    """
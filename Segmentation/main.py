import sys
import os

# Ajoutez le chemin du dossier parent au sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Dataloader import CustomDatasetWithLabels, CustomDataset, CustomDatasetWithLabelsFiltered
from Utils import filter_black_images_from_dataset
from Visualize import visualize_image


if __name__ == "__main__":
    folder = "../Training/"
    dataset = CustomDatasetWithLabelsFiltered(folder, is_training=True)
    dataset = filter_black_images_from_dataset(dataset)

    print(len(dataset))
    for i in range(120, 145):
        img, label = dataset[i]
        print(f"shape: {img.shape}")
        print(f"shape: {label.shape}")
        print(f"Visualisation de l'image {i+1}")
        visualize_image(dataset, i, "mages/")

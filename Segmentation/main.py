from dataloader import CustomDataset
from visualize import visualize_image
from utils import filter_black_images_from_dataset

if __name__ == "__main__":
    folder = "../Training/"
    dataset = CustomDataset(folder)
    dataset = filter_black_images_from_dataset(dataset)

    print(len(dataset))
    for i in range(120, 145):
        img, label = dataset[i]
        print(f"shape: {img.shape}")
        print(f"shape: {label.shape}")
        print(f"Visualisation de l'image {i+1}")
        visualize_image(dataset, i, "mages/")

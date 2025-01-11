import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model import UNet
from torch.utils.data import DataLoader
from Dataloader import CustomDatasetWithLabelsFiltered
from Metrics import calculate_dice_per_class
import numpy as np
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model, loader):
    accuracies = []
    dice_scores = {0: [], 1: [], 2: [], 3: []}  # Pour chaque classe

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            accuracies.append((preds == labels).float().mean().item())
            dice_score = calculate_dice_per_class(preds, labels, 4)
            for cls, score in dice_score:
                dice_scores[cls].append(score)

    return accuracies, dice_scores

def load_and_evaluate_models(folder_prefix, versions, test_loader):
    results = []
    weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #weights = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    for version in versions:
        folder = f"{folder_prefix}{version}"
        for weight in weights:
            print(f"{folder}/newsegmenter_{weight}.pth")
            model_path = f"{folder}/newsegmenter_{weight}.pth"
            model = UNet(1, 4).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            accuracies, dice_scores = evaluate_model(model, test_loader)
            for acc in accuracies:
                results.append({'Folder': folder_prefix, 'Weight': weight, 'Metric': 'Accuracy', 'Value': acc})
            for cls, scores in dice_scores.items():
                for score in scores:
                    results.append({'Folder': folder_prefix, 'Weight': weight, 'Metric': f'Dice_Class{cls}', 'Value': score})

    return results

def generate_box_plots(df, metric, output_folder):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Weight', y='Value', hue='Folder', data=df[df['Metric'] == metric])
    plt.title(f'Distribution des scores {metric}')
    plt.xlabel('Coefficient')
    plt.ylabel(f'Score {metric}')
    plt.legend(title="Type de dossier")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{metric}_boxplot.png")
    plt.close()

if __name__ == '__main__':
    # Configurations
    test_dataset = '../Training'
    versions = [1, 2, 2]
    folder_prefixes = ['segEVENV', 'segEVENandODDV', 'segEVENnoiseV', 'segMULTIV']
    output_folder = 'ResultsALLSEG'
    os.makedirs(output_folder, exist_ok=True)

    # Charger le dataset
    test_dataset = CustomDatasetWithLabelsFiltered(test_dataset, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Évaluer les modèles
    all_results = []
    for folder_prefix in folder_prefixes:
        results = load_and_evaluate_models(folder_prefix, versions, test_loader)
        all_results.extend(results)

    # Convertir en DataFrame
    results_df = pd.DataFrame(all_results)
    results_df.head(10)

    # Générer les box plots pour chaque métrique
    metrics = ['Accuracy', 'Dice_Class0', 'Dice_Class1', 'Dice_Class2', 'Dice_Class3']
    for metric in metrics:
        generate_box_plots(results_df, metric, output_folder)

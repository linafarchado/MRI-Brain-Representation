import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from model import UNet
from torch.utils.data import DataLoader
from Dataloader import CustomDatasetWithLabelsFiltered
from Metrics import calculate_dice_per_class
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_detailed(model, loader):
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

# base color red
base_color = "#8c0707"
new_models_color = "#5081c7"

def get_model_label(model_name):
    """
    Génère un label pour l'axe x :
    - Retourne 'segmentation' pour le modèle rouge.
    - Extrait le poids du nom pour les autres modèles.
    """
    if model_name == 'segmentation.pth':
        return 'No interpolated images'
    else:
        weight = model_name.split('_')[-1].replace('.pth', '')  # Extraire le poids
        return f'Coeff {weight}'

def visualize_sorted_dice_scores(df, cls, folder):
    # Calculer la médiane des scores pour chaque modèle
    medians = df.groupby('Model')[f'Dice_Class{cls}'].median().sort_values()
    
    # Déterminer l'ordre des modèles
    ordered_models = medians.index.tolist()
    
    # Spécifier les couleurs : rouge pour `segmentation.pth`, une autre couleur pour les autres
    palette = {model: base_color if model == 'segmentation.pth' else new_models_color for model in ordered_models}

    # Plot des scores Dice (boxplot)
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        x='Model',
        y=f'Dice_Class{cls}',
        data=df,
        order=ordered_models,
        palette=palette
    )
    plt.title(f'Distribution triée des scores Dice pour la classe {cls}')
    plt.xlabel('Model')
    plt.ylabel(f'Dice Score Class {cls}')
    
    # Mettre à jour les labels de l'axe x
    x_labels = [get_model_label(model) for model in ordered_models]
    plt.xticks(ticks=range(len(ordered_models)), labels=x_labels, rotation=65)
    
    plt.tight_layout()
    plt.savefig(f'{folder}/sorted_dice_scores_class{cls}.png')
    plt.close()

def visualize_sorted_accuracy(df, folder):
    # Calculer la médiane des précisions pour chaque modèle
    medians = df.groupby('Model')['Accuracy'].median().sort_values()
    
    # Déterminer l'ordre des modèles
    ordered_models = medians.index.tolist()
    
    # Spécifier les couleurs : rouge pour `segmentation.pth`, une autre couleur pour les autres
    palette = {model: base_color if model == 'segmentation.pth' else new_models_color for model in ordered_models}

    # Plot des accuracies (boxplot)
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        x='Model',
        y='Accuracy',
        data=df,
        order=ordered_models,
        palette=palette
    )
    plt.title('Distribution triée des précisions par modèle')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    
    # Mettre à jour les labels de l'axe x
    x_labels = [get_model_label(model) for model in ordered_models]
    plt.xticks(ticks=range(len(ordered_models)), labels=x_labels, rotation=65)
    
    plt.tight_layout()
    plt.savefig(f'{folder}/sorted_accuracies.png')
    plt.close()

def detailed_model_comparison(test_dataset, folder):
    test_dataset = CustomDatasetWithLabelsFiltered(test_dataset, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=1)

    models = {}
    
    segmenter_model_name = 'segmentation.pth'
    segmenter_model_instance = UNet(1, 4).to(device)
    segmenter_model_instance.load_state_dict(torch.load(segmenter_model_name, map_location=device))
    segmenter_model_instance.eval()
    models[segmenter_model_name] = segmenter_model_instance

    weights = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for i in weights:
        model_name = f'{folder}/newsegmenter_{i}.pth'
        model_instance = UNet(1, 4).to(device)
        model_instance.load_state_dict(torch.load(model_name, map_location=device))
        model_instance.eval()
        models[model_name] = model_instance

    # Évaluation
    all_accuracies = []
    all_dice_scores = {0: [], 1: [], 2: [], 3: []}
    model_names = []

    for model_name, model_instance in models.items():
        accuracies, dice_scores = evaluate_detailed(model_instance, test_loader)
        all_accuracies.extend(accuracies)
        for cls in all_dice_scores.keys():
            all_dice_scores[cls].extend(dice_scores[cls])
        model_names.extend([model_name] * len(accuracies))

    df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': all_accuracies,
        'Dice_Class0': all_dice_scores[0],
        'Dice_Class1': all_dice_scores[1],
        'Dice_Class2': all_dice_scores[2],
        'Dice_Class3': all_dice_scores[3]
    })

    # Visualiser les scores Dice triés par classe
    for cls in range(4):
        visualize_sorted_dice_scores(df, cls, folder)

    # Visualiser l'accuracy triée
    visualize_sorted_accuracy(df, folder)

if __name__ == '__main__':
    test_dataset = '../Training'
    folders = [
        'segEVENV2',
        'segEVENandODDV2',
        'segMULTIV2',
        'segEVENnoiseV2',
        'segEVENV1',
        'segEVENandODDV1',
        'segMULTIV1',
        'segEVENnoiseV1'
    ]
    
    for folder in folders:
        print(f'Comparing models in folder: {folder}')
        detailed_model_comparison(test_dataset, folder)

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import os
import json

# Ajoutez le chemin du dossier parent au sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from torch.utils.data import DataLoader
from Dataloader import CustomDatasetWithLabelsFiltered
from Metrics import calculate_dice_per_class
from model import UNet

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_min_max_dice(ordered_models, final_df, cls, dice_min_max_results):
    min_max_dice = {}
    for model in ordered_models:
        model_df = final_df[final_df['Model'] == model]
        min_max_dice[model] = {
            'min': model_df['Dice Score'].min(),
            'max': model_df['Dice Score'].max()
        }

    dice_min_max_results[f'Classe {cls}'] = min_max_dice
    return dice_min_max_results

def extract_min_max_accuracies(ordered_models, final_df):
    min_max_accuracies = {}
    for model in ordered_models:
        model_df = final_df[final_df['Model'] == model]
        min_max_accuracies[model] = {
            'min': model_df['Accuracy'].min(),
            'max': model_df['Accuracy'].max()
        }

    return min_max_accuracies

def process_group(group, test_dataset):
    test_dataset = CustomDatasetWithLabelsFiltered(test_dataset, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=1)

    weights = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    models = {}
    
    # Charger le modèle de base
    segmenter_model_name = 'segmentation.pth'
    segmenter_model_instance = UNet(1, 4).to(device)
    segmenter_model_instance.load_state_dict(torch.load(segmenter_model_name, map_location=device))
    segmenter_model_instance.eval()
    models[segmenter_model_name] = segmenter_model_instance

    # Générer des graphiques pour chaque classe
    dice_min_max_results = {}

    for cls in range(4):
        all_dice_scores_data = []

        # Évaluer le modèle de base
        _, base_dice_scores = evaluate_detailed(segmenter_model_instance, test_loader)
        base_df = pd.DataFrame({
            'Model': ['Base'] * len(base_dice_scores[cls]),
            'Dice Score': base_dice_scores[cls],
            'Weight': 0  # Pour le modèle de base
        })
        all_dice_scores_data.append(base_df)

        # Parcourir les différents poids
        for weight in weights:
            file_name = f'newsegmenter_{weight}.pth'
            models_names = [f'{group}V1NEW/{file_name}', f'{group}V2NEW/{file_name}', f'{group}V3NEW/{file_name}']
            
            # Combiner les Dice scores des 3 modèles pour ce poids
            combined_dice_scores = []
            
            for model_path in models_names:
                # Charger le modèle
                model_instance = UNet(1, 4).to(device)
                model_instance.load_state_dict(torch.load(model_path, map_location=device))
                model_instance.eval()

                # Évaluer le modèle
                _, dice_scores = evaluate_detailed(model_instance, test_loader)
                combined_dice_scores.extend(dice_scores[cls])
            
            # Créer un DataFrame pour ce poids
            weight_df = pd.DataFrame({
                'Model': [f'Weight {weight}'] * len(combined_dice_scores),
                'Dice Score': combined_dice_scores,
                'Weight': weight
            })
            all_dice_scores_data.append(weight_df)

        # Combiner tous les DataFrames
        final_df = pd.concat(all_dice_scores_data, ignore_index=True)

        # Calculer les médianes et trier les modèles
        medians = final_df.groupby('Model')['Dice Score'].median().sort_values()
        ordered_models = medians.index.tolist()

        # Extraire les valeurs min et max pour chaque modèle
        dice_min_max_results = extract_min_max_dice(ordered_models, final_df, cls, dice_min_max_results)

        """# Définir un schéma de couleurs personnalisé
        base_color = "#8c0707"
        new_models_color = "#5081c7"
        
        # Créer un dictionnaire de palette de couleurs
        palette = {model: base_color if model == 'Base' else new_models_color for model in ordered_models}

        # Créer le boxplot pour cette classe avec les modèles triés et colorés
        plt.figure(figsize=(15, 8))
        sns.boxplot(
            x='Model', 
            y='Dice Score', 
            data=final_df, 
            order=ordered_models,
            palette=palette
        )
        plt.title(f'Comparaison des Dice Scores pour la classe {cls} - Groupe {group}')
        plt.xlabel('Modèles')
        plt.ylabel('Dice Score')
        plt.xticks(rotation=65)
        plt.tight_layout()
        plt.savefig(f'{group}/dice_scores_class_{cls}_sorted.png')
        plt.close()"""

    return dice_min_max_results

def process_group_accuracies(group, test_dataset):
    test_dataset = CustomDatasetWithLabelsFiltered(test_dataset, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=1)

    weights = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    models = {}
    
    # Charger le modèle de base
    segmenter_model_name = 'segmentation.pth'
    segmenter_model_instance = UNet(1, 4).to(device)
    segmenter_model_instance.load_state_dict(torch.load(segmenter_model_name, map_location=device))
    segmenter_model_instance.eval()
    models[segmenter_model_name] = segmenter_model_instance

    all_accuracies_data = []

    # Évaluer le modèle de base
    base_accuracies, _ = evaluate_detailed(segmenter_model_instance, test_loader)
    base_df = pd.DataFrame({
        'Model': ['Base'] * len(base_accuracies),
        'Accuracy': base_accuracies,
        'Weight': 0  # Pour le modèle de base
    })
    all_accuracies_data.append(base_df)

    # Parcourir les différents poids
    for weight in weights:
        file_name = f'newsegmenter_{weight}.pth'
        models_names = [f'{group}V1NEW/{file_name}', f'{group}V2NEW/{file_name}', f'{group}V3NEW/{file_name}']
        
        # Combiner les accuracies des 3 modèles pour ce poids
        combined_accuracies = []
        
        for model_path in models_names:
            # Charger le modèle
            model_instance = UNet(1, 4).to(device)
            model_instance.load_state_dict(torch.load(model_path, map_location=device))
            model_instance.eval()

            # Évaluer le modèle
            accuracies, _ = evaluate_detailed(model_instance, test_loader)
            combined_accuracies.extend(accuracies)
        
        # Créer un DataFrame pour ce poids
        weight_df = pd.DataFrame({
            'Model': [f'Weight {weight}'] * len(combined_accuracies),
            'Accuracy': combined_accuracies,
            'Weight': weight
        })
        all_accuracies_data.append(weight_df)

    # Combiner tous les DataFrames
    final_df = pd.concat(all_accuracies_data, ignore_index=True)

    # Calculer les médianes et trier les modèles
    medians = final_df.groupby('Model')['Accuracy'].median().sort_values()
    ordered_models = medians.index.tolist()

    # Extraire les valeurs min et max pour chaque modèle
    min_max_accuracies = extract_min_max_accuracies(ordered_models, final_df)

    """# Définir un schéma de couleurs personnalisé
    base_color = "#8c0707"
    new_models_color = "#5081c7"
    
    # Créer un dictionnaire de palette de couleurs
    palette = {model: base_color if model == 'Base' else new_models_color for model in ordered_models}

    # Créer le boxplot
    plt.figure(figsize=(15, 8))
    sns.boxplot(
        x='Model', 
        y='Accuracy', 
        data=final_df, 
        order=ordered_models,
        palette=palette
    )
    plt.title(f'Comparaison des accuracies pour le groupe {group}')
    plt.xlabel('Modèles')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=65)
    plt.tight_layout()
    plt.savefig(f'{group}/accuracies.png')
    plt.close()"""
    return min_max_accuracies

def load_existing_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return {}

if __name__ == '__main__':
    test_dataset = '../Training'
    groups = [
        'segEVEN', 
        'segEVENandODD', 
        'segMULTI', 
        'segEVENnoise'
    ]


    min_max_accuracies_group = {}
    dice_min_max_results_group = {}

    # Charger les données existantes si les fichiers existent
    min_max_accuracies_file = 'min_max_accuracies_group.json'
    dice_min_max_results_file = 'dice_min_max_results_group.json'

    min_max_accuracies_group = load_existing_data(min_max_accuracies_file)
    dice_min_max_results_group = load_existing_data(dice_min_max_results_file)


    for group in groups:
        os.makedirs(group, exist_ok=True)
        min_max_accuracies_group[group] = process_group_accuracies(group, test_dataset)
        dice_min_max_results_group[group] = process_group(group, test_dataset)

    # Save it in file 
    with open('min_max_accuracies_group.json', 'w') as file:
        json.dump(min_max_accuracies_group, file, indent=4)
    
    with open('dice_min_max_results_group.json', 'w') as file:
        json.dump(dice_min_max_results_group, file, indent=4)

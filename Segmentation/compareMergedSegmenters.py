import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import UNet
from torch.utils.data import DataLoader
from Dataloader import CustomDatasetWithLabelsFiltered
from Metrics import calculate_dice_per_class

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_color = "#8c0707"  # Couleur pour le modèle de référence
new_models_color = "#5081c7"  # Couleur pour les moyennes des newsegmenters


def evaluate_model(model, test_loader):
    """
    Évalue un modèle et retourne les métriques (Dice scores et Accuracy).
    """
    model.eval()
    accuracies = []
    dice_scores = {cls: [] for cls in range(4)}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            accuracies.append((preds == labels).float().mean().item())
            dice_score = calculate_dice_per_class(preds, labels, 4)
            for cls, score in dice_score:
                dice_scores[cls].append(score)
    
    mean_accuracies = np.mean(accuracies)
    mean_dice_scores = {cls: np.mean(dice_scores[cls]) for cls in dice_scores}
    return mean_accuracies, mean_dice_scores


def evaluate_suffix(group_folders, suffix, test_loader):
    """
    Évalue tous les modèles avec un suffixe donné (ex: 0.05) dans un groupe de dossiers.
    """
    dice_scores = []
    accuracies = []

    for folder in group_folders:
        for weight_file in os.listdir(folder):
            if weight_file.endswith(f'_{suffix}.pth') and 'newsegmenter' in weight_file:
                model_path = os.path.join(folder, weight_file)
                print(model_path)
                model = UNet(1, 4).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                accuracy, dice = evaluate_model(model, test_loader)

                accuracies.append(accuracy)
                dice_scores.append(dice)

    # Moyenne des Dice scores et de l'accuracy
    mean_accuracy = np.mean(accuracies)
    mean_dice_scores = {cls: np.mean([dice[cls] for dice in dice_scores]) for cls in range(4)}

    return mean_accuracy, mean_dice_scores


def visualize_results(group_name, suffix_results, baseline_results, output_folder):
    """
    Génère des graphiques comparant les moyennes des résultats par suffixe aux résultats de référence.
    """
    suffixes = sorted(suffix_results.keys())
    mean_accuracies = [suffix_results[suffix][0] for suffix in suffixes]
    mean_dice_scores = {cls: [suffix_results[suffix][1][cls] for suffix in suffixes] for cls in range(4)}

    # Dice Scores
    plt.figure(figsize=(12, 8))
    for cls in range(4):
        plt.plot(suffixes, mean_dice_scores[cls], label=f'Classe {cls}', marker='o')
    plt.axhline(y=np.mean(list(baseline_results[1].values())), color=base_color, linestyle='--', label='Baseline (Dice moyen)')
    plt.title(f'Scores Dice Moyens - {group_name}')
    plt.xlabel('Suffixes des modèles')
    plt.ylabel('Dice Score Moyen')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{group_name}_dice_scores.png'))
    plt.close()

    # Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(suffixes, mean_accuracies, label='Newsegmenters (Mean)', color=new_models_color, marker='o')
    plt.axhline(y=baseline_results[0], color=base_color, linestyle='--', label='Baseline (Accuracy)')
    plt.title(f'Accuracy Moyenne - {group_name}')
    plt.xlabel('Suffixes des modèles')
    plt.ylabel('Accuracy Moyenne')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{group_name}_accuracy.png'))
    plt.close()

def visualize_results_boxplot(group_name, suffix_results, baseline_results, output_folder):
    suffixes = sorted(suffix_results.keys())  # Liste triée des suffixes
    output_path = os.path.join(output_folder, f'{group_name}_results_boxplot.png')

    print(f"Traitement du groupe : {group_name}")  # Debugging

    # Vérification des données dans suffix_results et baseline_results
    print(f"Structure des données dans suffix_results et baseline_results :")
    for suffix in suffixes:
        print(f"Suffixe {suffix} - Contenu : {suffix_results[suffix]}")  # Debugging
    print(f"Baseline Results - Contenu : {baseline_results}")

    all_data = []
    for cls in range(4):  # Pour chaque classe
        try:
            class_data = [suffix_results[suffix][1][cls] for suffix in suffixes]
            class_data.append(baseline_results[1][cls])  # Ajouter les résultats baseline
            all_data.append(class_data)

            # Debugging : Vérifions les données extraites
            print(f"Classe {cls} - Suffixe Results : {[suffix_results[suffix][1][cls] for suffix in suffixes]}")
            print(f"Classe {cls} - Baseline Results : {baseline_results[1][cls]}")
        except Exception as e:
            print(f"Erreur pour la classe {cls} : {e}")
            continue

    # Création des positions (une pour chaque suffixe + baseline)
    positions = list(range(len(suffixes) + 1))  # Nombre de suffixes + 1 pour baseline
    print(f"Positions utilisées : {positions}")  # Debugging

    # Vérification de la correspondance entre données et positions
    if len(all_data[0]) != len(positions):
        print(f"ERREUR : len(data) = {len(all_data[0])}, len(positions) = {len(positions)}")
        raise ValueError("Le nombre de données pour le boxplot et les positions ne correspondent pas.")

    # Création du boxplot pour chaque classe
    plt.figure(figsize=(12, 8))
    for cls in range(4):
        plt.boxplot(
            all_data[cls],
            positions=positions,
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='blue'),
            flierprops=dict(marker='o', color='red', alpha=0.5),
            labels=[f"{suffix}" for suffix in suffixes] + ['Baseline']
        )

        plt.title(f'Boxplot des Dice Scores pour {group_name} - Classe {cls}')
        plt.xlabel('Modèles')
        plt.ylabel('Dice Score')
        plt.xticks(rotation=45)
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Boxplot sauvegardé : {output_path}")


def process_group(group_name, group_folders, test_dataset):
    """
    Traite un groupe de dossiers indépendants et génère des graphiques.
    """
    test_dataset = CustomDatasetWithLabelsFiltered(test_dataset, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Charger le modèle de référence
    baseline_model = UNet(1, 4).to(device)
    print('Loading baseline model...')
    if os.path.exists('segmentation.pth'):
        print('Loading model from segmentation.pth')
    baseline_model.load_state_dict(torch.load('segmentation.pth', map_location=device))
    baseline_results = evaluate_model(baseline_model, test_loader)

    # Calculer les métriques moyennes pour chaque suffixe
    #suffixes = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    suffixes = [0.05, 0.1]

    suffix_results = {}
    for suffix in suffixes:
        suffix_results[suffix] = evaluate_suffix(group_folders, suffix, test_loader)

    # Visualiser les résultats
    output_folder = os.path.join('Results', group_name)
    os.makedirs(output_folder, exist_ok=True)  # Crée un sous-dossier pour chaque groupe dans "Results/"
    visualize_results_boxplot(group_name, suffix_results, baseline_results, output_folder)


if __name__ == '__main__':
    test_dataset = '../Training'
    
    # Groupes définis sans dossier parent
    groups = {
        'segEVEN': ['segEVENV1NEW', 'segEVENV2NEW', 'segEVENV3NEW'],
        'segEVENandODD': ['segEVENandODDV1NEW', 'segEVENandODDV2NEW', 'segEVENandODDV3NEW'],
        'segMULTI': ['segMULTIV1NEW', 'segMULTIV2NEW', 'segMULTIV3NEW'],
        'segEVENnoise': ['segEVENnoiseV1NEW', 'segEVENnoiseV2NEW', 'segEVENnoiseV3NEW'],
    }

    # Boucle sur les groupes pour traiter chaque ensemble de dossiers
    for group_name, group_folders in groups.items():
        process_group(group_name, group_folders, test_dataset)

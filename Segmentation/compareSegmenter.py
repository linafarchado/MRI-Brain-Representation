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

def visualize_dice_scores(df):
    df_melted = pd.melt(df, id_vars=['Model'], value_vars=['Dice_Class0', 'Dice_Class1', 'Dice_Class2', 'Dice_Class3'],
                        var_name='Class', value_name='Dice Score')
    sns.boxplot(x='Class', y='Dice Score', hue='Model', data=df_melted)
    plt.title('Distribution des scores Dice par classe et par modèle')
    plt.savefig('dice_scores.png')

def detailed_model_comparison(test_dataset, folder):
    test_dataset = CustomDatasetWithLabelsFiltered(test_dataset, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=1)

    models = {}
    
    segmenter_model_name = 'segmentation.pth'
    segmenter_model_instance = UNet(1, 4).to(device)
    segmenter_model_instance.load_state_dict(torch.load(segmenter_model_name, map_location=device))
    segmenter_model_instance.eval()
    models[segmenter_model_name] = segmenter_model_instance


    #weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    weights = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for i in weights:
        model_name = f'{folder}/newsegmenter_{i}.pth'
        model_instance = UNet(1, 4).to(device)
        model_instance.load_state_dict(torch.load(model_name, map_location=device))
        model_instance.eval()
        models[model_name] = model_instance

    # Evaluation
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

    # Visualize dice scores
    for cls in range(4):
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Model', y=f'Dice_Class{cls}', data=df)
        plt.title(f'Distribution des scores Dice pour la classe {cls}')
        plt.xlabel('Model')
        plt.ylabel(f'{folder}/Dice Score Class {cls}')
        plt.xticks(rotation=65)
        plt.tight_layout()
        plt.savefig(f'{folder}/dice_scores_class{cls}.png')
        plt.close()

    # visualize accuracy
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Model', y='Accuracy', data=df)
    plt.title('Distribution des précisions par modèle')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=65)
    plt.tight_layout()
    plt.savefig(f'{folder}/accuracies.png')
    plt.close()

if __name__ == '__main__':
    test_dataset ='../Training'

    folder = [
        'segEVENV2',
        'segEVENandODDV2',
        'segMULTIV2',
        'segEVENnoiseV2',
        'segEVENV1',
        'segEVENandODDV1',
        'segMULTIV1',
        'segEVENnoiseV1'
        ]
    
    for f in range(len(folder)):
        detailed_model_comparison(test_dataset, folder[f])


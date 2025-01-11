import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys
import os

# Ajoutez le chemin du dossier parent au sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from torch.utils.data import DataLoader
from Dataloader import CustomDatasetWithLabelsFiltered
from Metrics import calculate_dice_per_class
from model import UNet

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

def plot_model_performance(files, device):
    test_dataset = CustomDatasetWithLabelsFiltered(test_dataset, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Prepare data structures to store results
    all_accuracies = []
    all_dice_scores = {0: [], 1: [], 2: [], 3: []}
    
    # Evaluate each model
    for file in files:
        # Load the model
        model = UNet(1, 4).to(device)
        model.load_state_dict(torch.load(file, map_location=device))
        model.eval()
        
        # Evaluate the model
        accuracies, dice_scores = evaluate_detailed(model, test_loader)
        all_accuracies.extend(accuracies)
        for cls in range(4):
            all_dice_scores[cls].extend(dice_scores[cls])
    
    # Prepare data for plotting
    plt.figure(figsize=(15, 10))
    
    # Accuracy subplot
    plt.subplot(2, 3, 1)
    plt.title('Accuracy')
    sns.boxplot(all_accuracies)
    plt.xticks([0], ['Accuracy'])
    plt.ylabel('Accuracy')
    
    # Dice Score subplots
    for i, cls in enumerate(range(4)):
        plt.subplot(2, 3, i+2)
        plt.title(f'Dice Score - Class {cls}')
        sns.boxplot(all_dice_scores[cls])
        plt.xticks([0], [f'Dice Score - Class {cls}'])
        plt.ylabel('Dice Score')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('model_performance.png')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = '../Training'
    file = "newsegmenter_0.05.pth"
    files = [
        f'segEVENV1NEW/{file}',
        f'segEVENV2NEW/{file}',
        f'segEVENV3NEW/{file}'
    ]
    
    
    plot_model_performance(files, device)
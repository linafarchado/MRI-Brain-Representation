import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from model import UNet
from torch.utils.data import DataLoader
from Dataloader import CustomDatasetWithLabelsFiltered
from Metrics import calculate_dice_per_class
import os
from matplotlib.patches import Patch

def evaluate_detailed(model, loader, device):
    accuracies = []
    dice_scores = {0: [], 1: [], 2: [], 3: []}
    
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

def create_comprehensive_boxplots(dataframes, comparisionfolder):
    os.makedirs(comparisionfolder, exist_ok=True)
    metrics = ['Accuracy', 'Dice_Class0', 'Dice_Class1', 'Dice_Class2', 'Dice_Class3']
    metric_names = ['Accuracy', 'Dice Score Class 0', 'Dice Score Class 1', 'Dice Score Class 2', 'Dice Score Class 3']
    
    base_color = "#8c0707"
    new_models_color = "#5081c7"

    # Mapping des dossiers vers des labels lisibles
    folder_labels = {
        'segEVENandODDV2': 'Even and Odd',
        'segMULTIV2': '4 Images',
        'segEVENnoiseV2': 'Noise',
        'segEVENV1': 'Even'
    }
    
    for metric, metric_name in zip(metrics, metric_names):
        plt.figure(figsize=(16, 10))
        
        # Collect data for boxplots
        boxplot_data = [dataframes['base_model'][metric]]
        folder_names = ['Base Model']
        
        for folder, df in dataframes['folder_models'].items():
            boxplot_data.append(df[metric])
            folder_names.append(folder_labels.get(folder, folder))

        # Create boxplot
        bp = plt.boxplot(boxplot_data, patch_artist=True, widths=0.6)
        
        # Color the boxes
        plt.setp(bp['boxes'][0], facecolor=base_color)
        for box in bp['boxes'][1:]:
            plt.setp(box, facecolor=new_models_color)
        
        # Black median lines
        for median in bp['medians']:
            median.set(color='black', linewidth=2)
        
        # Add median values
        for i, median in enumerate(bp['medians']):
            median_value = median.get_ydata()[0]
            plt.text(i+1, median_value, f'{median_value:.4f}', 
                     horizontalalignment='center', verticalalignment='bottom', 
                     fontsize=10, color='black')
        
        # Customize plot
        plt.title(f'Distribution of {metric_name} Across Models', fontsize=16)
        plt.xlabel('Models', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.xticks(range(1, len(folder_names)+1), folder_names, ha='center')
        
        # Create legend
        legend_elements = [
            Patch(facecolor=base_color, label='Base Model'),
            Patch(facecolor=new_models_color, label='New Models')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{comparisionfolder}/{metric}_comparison.png')
        plt.close()

def detailed_model_comparison(test_dataset, folders):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = CustomDatasetWithLabelsFiltered(test_dataset, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Collect all results
    results = {}
    
    # Base model
    segmenter_model_name = 'segmentation.pth'
    segmenter_model = UNet(1, 4).to(device)
    segmenter_model.load_state_dict(torch.load(segmenter_model_name, map_location=device))
    segmenter_model.eval()
    
    base_accuracies, base_dice_scores = evaluate_detailed(segmenter_model, test_loader, device)
    base_df = pd.DataFrame({
        'Accuracy': base_accuracies,
        'Dice_Class0': base_dice_scores[0],
        'Dice_Class1': base_dice_scores[1],
        'Dice_Class2': base_dice_scores[2],
        'Dice_Class3': base_dice_scores[3]
    })
    
    # Results for all folders
    folder_results = {}
    
    for folder in folders:
        print(f'Processing folder: {folder}')

        models = {}
        weights = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        for w in weights:
            model_name = f'{folder}/newsegmenter_{w}.pth'
            model = UNet(1, 4).to(device)
            model.load_state_dict(torch.load(model_name, map_location=device))
            model.eval()
            models[model_name] = model
        
        all_data = []
        for model_name, model in models.items():
            accuracies, dice_scores = evaluate_detailed(model, test_loader, device)
            
            for i in range(len(accuracies)):
                row_data = {
                    'Accuracy': accuracies[i],
                    'Dice_Class0': dice_scores[0][i],
                    'Dice_Class1': dice_scores[1][i],
                    'Dice_Class2': dice_scores[2][i],
                    'Dice_Class3': dice_scores[3][i]
                }
                all_data.append(row_data)
        
        folder_df = pd.DataFrame(all_data)
        folder_results[folder] = folder_df
    
    # Prepare data for comprehensive plotting
    comprehensive_data = {
        'base_model': base_df,
        'folder_models': folder_results
    }
    
    # Create comprehensive boxplots
    print('Creating comprehensive boxplots...')
    create_comprehensive_boxplots(comprehensive_data, 'SegmentersComparison')

if __name__ == '__main__':
    test_dataset = '../Training'
    folders = [
        'segEVENV1',
        'segEVENandODDV2',
        'segMULTIV2',
        'segEVENnoiseV2',
    ]
    
    detailed_model_comparison(test_dataset, folders)
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from model import UNet
from torch.utils.data import DataLoader
from Dataloader import CustomDatasetWithLabelsFiltered
from Metrics import calculate_dice_per_class

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

def create_enhanced_boxplots(df, folder, metric_columns):
    # Colors for different categories
    base_color = "#8c0707"
    new_models_color = "#5081c7"

    for metric in metric_columns:
        plt.figure(figsize=(12, 8))
        
        # Separate data for different model types
        base_model_data = df[df['Model'] == 'segmentation.pth'][metric]
        new_models_data = df[df['Model'] != 'segmentation.pth'][metric]
        
        # Create positions for boxes
        positions = np.arange(2)
        
        # Create boxplots
        bp1 = plt.boxplot(base_model_data, positions=[positions[0]], 
                         patch_artist=True, widths=0.6)
        bp2 = plt.boxplot(new_models_data, positions=[positions[1]], 
                         patch_artist=True, widths=0.6)
        
        # Color the boxes
        plt.setp(bp1['boxes'], facecolor=base_color)
        plt.setp(bp2['boxes'], facecolor=new_models_color)

        # Force the median color to black for both boxplots
        for median in bp1['medians']:
            median.set(color='black')  # Black median, thicker line
        for median in bp2['medians']:
            median.set(color='black')  # Black median, thicker line
        
        # Add median values as text on the boxplot
        for median, position in zip(bp1['medians'], [positions[0]]):
            median_value = median.get_ydata()[0]
            plt.text(position, median_value, f'{median_value:.4f}', 
                     horizontalalignment='center', verticalalignment='bottom', fontsize=10, color='black')
        
        for median, position in zip(bp2['medians'], [positions[1]]):
            median_value = median.get_ydata()[0]
            plt.text(position, median_value, f'{median_value:.4f}', 
                     horizontalalignment='center', verticalalignment='bottom', fontsize=10, color='black')
        

        # Customize plot
        plt.xticks(positions, ['Base Model\n(segmentation.pth)', 
                             'New Models'])
        plt.title(f'Distribution of {metric} Scores')
        plt.ylabel(metric)
        
        legend_elements = [
            Patch(facecolor=base_color, label='Base Model'),
            Patch(facecolor=new_models_color, label='New Models')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{folder}/{metric}_enhanced_comparison.png')
        plt.close()

def detailed_model_comparison(test_dataset, folder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = CustomDatasetWithLabelsFiltered(test_dataset, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Initialize models dictionary
    models = {}
    
    # Load base model
    segmenter_model_name = 'segmentation.pth'
    segmenter_model = UNet(1, 4).to(device)
    segmenter_model.load_state_dict(torch.load(segmenter_model_name, map_location=device))
    segmenter_model.eval()
    models[segmenter_model_name] = segmenter_model
    
    # Load new models
    weights = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for w in weights:
        model_name = f'{folder}/newsegmenter_{w}.pth'
        model = UNet(1, 4).to(device)
        model.load_state_dict(torch.load(model_name, map_location=device))
        model.eval()
        models[model_name] = model
    
    # Evaluation
    all_data = []
    image_index = 0
    
    for model_name, model in models.items():
        accuracies, dice_scores = evaluate_detailed(model, test_loader, device)
        
        for i in range(len(accuracies)):
            row_data = {
                'Model': model_name,
                'Image_Index': image_index,
                'Accuracy': accuracies[i],
                'Dice_Class0': dice_scores[0][i],
                'Dice_Class1': dice_scores[1][i],
                'Dice_Class2': dice_scores[2][i],
                'Dice_Class3': dice_scores[3][i]
            }
            all_data.append(row_data)
            image_index += 1
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Create enhanced boxplots for each metric
    metric_columns = ['Accuracy', 'Dice_Class0', 'Dice_Class1', 'Dice_Class2', 'Dice_Class3']
    create_enhanced_boxplots(df, folder, metric_columns)

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
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

def detailed_model_comparison(test_dataset):
    # Configuration
    test_dataset = CustomDatasetWithLabelsFiltered(test_dataset, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Chargement des modèles
    model = UNet(1, 4).to(device)
    model.load_state_dict(torch.load('segmentation.pth', map_location=device))
    model.eval()

    model0_4 = UNet(1, 4).to(device)
    model0_4.load_state_dict(torch.load('newsegmenter_0.4.pth', map_location=device))
    model0_4.eval()

    model0_5 = UNet(1, 4).to(device)
    model0_5.load_state_dict(torch.load('newsegmenter_0.5.pth', map_location=device))
    model0_5.eval()

    model0_7 = UNet(1, 4).to(device)
    model0_7.load_state_dict(torch.load('newsegmenter_0.7.pth', map_location=device))
    model0_7.eval()

    model0_9 = UNet(1, 4).to(device)
    model0_9.load_state_dict(torch.load('newsegmenter_0.9.pth', map_location=device))
    model0_9.eval()

    # Évaluation des modèles
    accuracies, dice_scores = evaluate_detailed(model, test_loader)
    accuracies0_4, dice_scores0_4 = evaluate_detailed(model0_4, test_loader)
    accuracies0_5, dice_scores0_5 = evaluate_detailed(model0_5, test_loader)
    accuracies0_7, dice_scores0_7 = evaluate_detailed(model0_7, test_loader)
    accuracies0_9, dice_scores0_9 = evaluate_detailed(model0_9, test_loader)

    # Création d'un DataFrame pour faciliter la visualisation
    df = pd.DataFrame({
        'Model': ['segmentation.pth'] * len(accuracies) + ['newsegmenter_0.4.pth'] * len(accuracies0_4) + ['newsegmenter_0.5.pth'] * len(accuracies0_5) + ['newsegmenter_0.7.pth'] * len(accuracies0_7) + ['newsegmenter_0.9.pth'] * len(accuracies0_9),
        'Accuracy': accuracies + accuracies0_4 + accuracies0_5 + accuracies0_7 + accuracies0_9,
        'Dice_Class0': dice_scores[0] + dice_scores0_4[0] + dice_scores0_5[0] + dice_scores0_7[0] + dice_scores0_9[0],
        'Dice_Class1': dice_scores[1] + dice_scores0_4[1] + dice_scores0_5[1] + dice_scores0_7[1] + dice_scores0_9[1],
        'Dice_Class2': dice_scores[2] + dice_scores0_4[2] + dice_scores0_5[2] + dice_scores0_7[2] + dice_scores0_9[2],
        'Dice_Class3': dice_scores[3] + dice_scores0_4[3] + dice_scores0_5[3] + dice_scores0_7[3] + dice_scores0_9[3]
    })

    visualize_dice_scores(df)
    
    """

    # Visualisations détaillées
    plt.figure(figsize=(20, 15))
    
    # 1. Distribution des précisions
    plt.subplot(2, 2, 1)
    sns.boxplot(x='Model', y='Accuracy', data=df)
    plt.title('Distribution des précisions par modèle')
    
    # 2. Distribution des scores Dice par classe
    plt.subplot(2, 2, 2)
    df_melted = pd.melt(df, id_vars=['Model'], value_vars=['Dice_Class0', 'Dice_Class1', 'Dice_Class2', 'Dice_Class3'],
                        var_name='Class', value_name='Dice Score')
    sns.boxplot(x='Class', y='Dice Score', hue='Model', data=df_melted)
    plt.title('Distribution des scores Dice par classe et par modèle')
    
    # 3. Courbe de densité des précisions
    plt.subplot(2, 2, 3)
    sns.kdeplot(data=df, x='Accuracy', hue='Model', shade=True)
    plt.title('Densité des précisions par modèle')
    
    # 4. Heatmap des corrélations entre métriques
    plt.subplot(2, 2, 4)
    # Exclude the 'Model' column for correlation calculation
    numeric_df = df.drop('Model', axis=1)
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Corrélations entre les métriques')
    
    plt.tight_layout()
    plt.savefig('detailed_model_comparison.png')
    plt.close()
    
    # Affichage des statistiques globales
    print(df.groupby('Model').agg({
        'Accuracy': ['mean', 'std'],
        'Dice_Class0': ['mean', 'std'],
        'Dice_Class1': ['mean', 'std'],
        'Dice_Class2': ['mean', 'std'],
        'Dice_Class3': ['mean', 'std']
    }))"""

if __name__ == '__main__':
    test_dataset ='../Training'
    detailed_model_comparison(test_dataset)


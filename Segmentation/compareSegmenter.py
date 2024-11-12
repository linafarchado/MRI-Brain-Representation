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

def visualize_dice_scores_bis(df):
    df_melted = pd.melt(df, id_vars=['Model'], value_vars=['Dice_Class0', 'Dice_Class1', 'Dice_Class2', 'Dice_Class3'],
                        var_name='Class', value_name='Dice Score')
    
    # Create a box plot using matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a dictionary to store box plot data
    box_data = {}
    for model in df_melted['Model'].unique():
        box_data[model] = [df_melted[(df_melted['Model'] == model) & (df_melted['Class'] == cls)]['Dice Score'].values 
                           for cls in ['Dice_Class0', 'Dice_Class1', 'Dice_Class2', 'Dice_Class3']]
    
    # Plot each model's data
    positions = np.arange(len(box_data['segmenter.pth'])) * (len(box_data) + 1)
    for i, (model, data) in enumerate(box_data.items()):
        bp = ax.boxplot(data, positions=positions + i, widths=0.6, patch_artist=True, 
                        boxprops=dict(facecolor='C'+str(i), color='C'+str(i)),
                        medianprops=dict(color='black'))
    
    # Set x-ticks and labels
    ax.set_xticks(positions + (len(box_data) - 1) / 2)
    ax.set_xticklabels(['Class 0', 'Class 1', 'Class 2', 'Class 3'])
    
    # Add legend
    handles = [plt.Line2D([0], [0], color='C'+str(i), lw=4) for i in range(len(box_data))]
    ax.legend(handles, box_data.keys(), title='Model')
    
    # Set titles and labels
    ax.set_title('Distribution des scores Dice par classe et par modèle')
    ax.set_xlabel('Class')
    ax.set_ylabel('Dice Score')
    
    plt.savefig('dice_scores_bis.png')
    plt.close()

def detailed_model_comparison(test_dataset):
    # Configuration
    test_dataset = CustomDatasetWithLabelsFiltered(test_dataset, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=1)

    # Chargement des modèles
    """    
    model = UNet(1, 4).to(device)
    model.load_state_dict(torch.load('segmentation.pth', map_location=device))
    model.eval()

    model0_0_5 = UNet(1, 4).to(device)
    model0_0_5.load_state_dict(torch.load('newsegmenter_0.05.pth', map_location=device))
    model0_0_5.eval()

    model0_1 = UNet(1, 4).to(device)
    model0_1.load_state_dict(torch.load('newsegmenter_0.1.pth', map_location=device))
    model0_1.eval()

    model0_1_5 = UNet(1, 4).to(device)
    model0_1_5.load_state_dict(torch.load('newsegmenter_0.15.pth', map_location=device))
    model0_1_5.eval()

    model0_2 = UNet(1, 4).to(device)
    model0_2.load_state_dict(torch.load('newsegmenter_0.2.pth', map_location=device))
    model0_2.eval()

    model0_2_5 = UNet(1, 4).to(device)
    model0_2_5.load_state_dict(torch.load('newsegmenter_0.25.pth', map_location=device))
    model0_2_5.eval()"""

    models = {}
    
    # Charger le modèle segmenter.pth
    segmenter_model_name = 'segmentation.pth'
    segmenter_model_instance = UNet(1, 4).to(device)
    segmenter_model_instance.load_state_dict(torch.load(segmenter_model_name, map_location=device))
    segmenter_model_instance.eval()
    models[segmenter_model_name] = segmenter_model_instance

    weights = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.85, 0.9, 0.95]
    for i in weights:
        model_name = f'newsegmenter_{i}.pth'
        model_instance = UNet(1, 4).to(device)
        model_instance.load_state_dict(torch.load(model_name, map_location=device))
        model_instance.eval()
        models[model_name] = model_instance

    # Évaluation des modèles
    all_accuracies = []
    all_dice_scores = {0: [], 1: [], 2: [], 3: []}
    model_names = []

    for model_name, model_instance in models.items():
        accuracies, dice_scores = evaluate_detailed(model_instance, test_loader)
        all_accuracies.extend(accuracies)
        for cls in all_dice_scores.keys():
            all_dice_scores[cls].extend(dice_scores[cls])
        model_names.extend([model_name] * len(accuracies))

    # Création d'un DataFrame pour faciliter la visualisation
    df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': all_accuracies,
        'Dice_Class0': all_dice_scores[0],
        'Dice_Class1': all_dice_scores[1],
        'Dice_Class2': all_dice_scores[2],
        'Dice_Class3': all_dice_scores[3]
    })

    # Visualisation des scores Dice par classe
    for cls in range(4):
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='Model', y=f'Dice_Class{cls}', data=df)
        plt.title(f'Distribution des scores Dice pour la classe {cls}')
        plt.xlabel('Model')
        plt.ylabel(f'Dice Score Class {cls}')
        plt.xticks(rotation=65)
        plt.tight_layout()
        plt.savefig(f'dice_scores_class{cls}.png')
        plt.close()

    # visualize accuracy
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Model', y='Accuracy', data=df)
    plt.title('Distribution des précisions par modèle')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=65)
    plt.tight_layout()
    plt.savefig('accuracies.png')
    plt.close()

    """# Évaluation des modèles
    accuracies, dice_scores = evaluate_detailed(model, test_loader)
    accuracies0_0_5, dice_scores0_0_5 = evaluate_detailed(model0_0_5, test_loader)
    accuracies0_1, dice_scores0_1 = evaluate_detailed(model0_1, test_loader)
    accuracies0_1_5, dice_scores0_1_5 = evaluate_detailed(model0_1_5, test_loader)
    accuracies0_2, dice_scores0_2 = evaluate_detailed(model0_2, test_loader)
    accuracies0_2_5, dice_scores0_2_5 = evaluate_detailed(model0_2_5, test_loader)

    # Création d'un DataFrame pour faciliter la visualisation
    df = pd.DataFrame({
        'Model': ['segmenter.pth'] * len(accuracies) + ['newsegmenter_0.05.pth'] * len(accuracies0_0_5) + ['newsegmenter_0.1.pth'] * len(accuracies0_1) + ['newsegmenter_0.15.pth'] * len(accuracies0_1_5) + ['newsegmenter_0.2.pth'] * len(accuracies0_2) + ['newsegmenter_0.25.pth'] * len(accuracies0_2_5),
        'Accuracy': accuracies + accuracies0_0_5 + accuracies0_1 + accuracies0_1_5 + accuracies0_2 + accuracies0_2_5,
        'Dice_Class0': dice_scores[0] + dice_scores0_0_5[0] + dice_scores0_1[0] + dice_scores0_1_5[0] + dice_scores0_2[0] + dice_scores0_2_5[0],
        'Dice_Class1': dice_scores[1] + dice_scores0_0_5[1] + dice_scores0_1[1] + dice_scores0_1_5[1] + dice_scores0_2[1] + dice_scores0_2_5[1],
        'Dice_Class2': dice_scores[2] + dice_scores0_0_5[2] + dice_scores0_1[2] + dice_scores0_1_5[2] + dice_scores0_2[2] + dice_scores0_2_5[2],
        'Dice_Class3': dice_scores[3] + dice_scores0_0_5[3] + dice_scores0_1[3] + dice_scores0_1_5[3] + dice_scores0_2[3] + dice_scores0_2_5[3]
    })

    visualize_dice_scores(df)
    visualize_dice_scores_bis(df)
    print(f'Segmentation.pth accuracy: {np.mean(accuracies):.4f}')
    print(f'Newsegmenter_0.05.pth accuracy: {np.mean(accuracies0_0_5):.4f}')
    print(f'Newsegmenter_0.1.pth accuracy: {np.mean(accuracies0_1):.4f}')
    print(f'Newsegmenter_0.15.pth accuracy: {np.mean(accuracies0_1_5):.4f}')
    print(f'Newsegmenter_0.2.pth accuracy: {np.mean(accuracies0_2):.4f}')
    print(f'Newsegmenter_0.25.pth accuracy: {np.mean(accuracies0_2_5):.4f}')
    
    """
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


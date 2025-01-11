import sys
import os
import json
import pandas as pd

# Ajoutez le chemin du dossier parent au sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Dataloader import CustomDatasetWithLabels, CustomDataset, CustomDatasetWithLabelsFiltered
from Utils import filter_black_images_from_dataset, process_seg_data, process_seg_accuracy_data, process_base_values, process_base_values_dice
from Visualize import visualize_image

def load_json_file(filename):
    """Load JSON data from a file."""
    with open(filename, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    filename = 'dice_min_max_results_group.json'
    #filename = "min_max_accuracies_group.json"

    df = pd.DataFrame()
    data = load_json_file(filename)

    groups = [
        'segEVEN',
        #'segEVENandODD',
        #'segMULTI',
        #'segEVENnoise'
    ]
    for group in groups:
        #df = process_seg_data(data, group)
        df = process_base_values_dice(data, group)
        print(df)
        
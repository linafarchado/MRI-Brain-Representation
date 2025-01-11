import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def load_json_file(filename):
    """Load JSON data from a file."""
    with open(filename, 'r') as file:
        return json.load(file)

def create_statistics_dataframe(filename):
    metric = "Dice" if "dice" in filename else "Accuracy"

    # Load JSON data
    json_data = load_json_file(filename)
    
    # Groups to analyze
    groups = [
        'segEVEN',
        'segEVENandODD', 
        'segMULTI', 
        'segEVENnoise'
    ]
    
    # Prepare lists to store data
    model_names = []
    min_mins = []
    min_maxs = []
    min_bases = []
    max_mins = []
    max_maxs = []
    max_bases = []
    
    # Collect data for each group
    for group in groups:
        group_data = json_data[group]
        
        # Skip 'Base' entry when collecting data
        filtered_data = {k: v for k, v in group_data.items() if k != "Base"}
        
        # Base model values
        base_min = group_data['Base']['min']
        base_max = group_data['Base']['max']
        
        # Find extreme values
        mins = [entry['min'] for entry in filtered_data.values()]
        maxs = [entry['max'] for entry in filtered_data.values()]
        
        # Extreme values
        min_min = min(mins)
        min_max = max(mins)
        max_min = min(maxs)
        max_max = max(maxs)
        
        # Store data
        model_names.append(group)
        min_mins.append(min_min)
        min_maxs.append(min_max)
        min_bases.append(base_min)
        max_mins.append(max_min)
        max_maxs.append(max_max)
        max_bases.append(base_max)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Model': model_names,
        'min_min': min_mins,
        'min_max': min_maxs,
        'min_base': min_bases,
        'max_min': max_mins,
        'max_max': max_maxs,
        'max_base': max_bases
    })
    
    # Calculate differences and additional columns
    df['min_diff_low'] = df['min_base'] - df['min_min']
    df['min_diff_high'] = df['min_max'] - df['min_base']
    df['max_diff_low'] = df['max_base'] - df['max_min']
    df['max_diff_high'] = df['max_max'] - df['max_base']
    
    visualize_dataframe(df, groups, metric)
    return df

def visualize_dataframe(df, groups, metric):
    plt.figure(figsize=(12, 6))
    
    # Bar width and positions
    width = 0.2
    x = np.arange(len(groups))
    
    # Plot bars
    plt.bar(x - 1.5*width, df['min_diff_high'], width, label='Highest Min Diff', color='blue')
    plt.bar(x - 0.5*width, df['min_diff_low'], width, label='Lowest Min Diff', color='lightblue')
    
    # Customize plot
    plt.xlabel('Groups')
    plt.ylabel('Difference from Base Model')
    plt.title('Variations in Min Values Across Groups')
    plt.xticks(x, groups)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{metric}_group_variations_min.png')
    plt.close()

    # Plotting
    plt.figure(figsize=(12, 6))

    # Bar width and positions
    width = 0.2
    x = np.arange(len(groups))

    plt.bar(x + 0.5*width, df['max_diff_high'], width, label='Highest Max Diff', color='red')
    plt.bar(x + 1.5*width, df['max_diff_low'], width, label='Lowest Max Diff', color='lightcoral')

    # Customize plot
    plt.xlabel('Groups')
    plt.ylabel('Difference from Base Model')
    plt.title('Variations in Max Values Across Groups')
    plt.xticks(x, groups)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{metric}_group_variations_max.png')
    plt.close()

def main():
    # Filename of the JSON data
    filename = 'min_max_accuracies_group.json'
    
    # Create DataFrame
    df = create_statistics_dataframe(filename)
    
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
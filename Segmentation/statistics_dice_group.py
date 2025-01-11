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
    metric = "Dice"

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
    results = []
    
    # Collect data for each group
    for group in groups:
        group_data = json_data[group]
        
        # Analyze each class in the group
        for classe in [key for key in group_data.keys() if key != 'Base']:
            classe_data = group_data[classe]
            
            # Skip 'Base' entry when collecting data
            filtered_data = {k: v for k, v in classe_data.items() if k != "Base"}
            
            # Base model values
            base_min = classe_data['Base']['min']
            base_max = classe_data['Base']['max']
            
            # Find extreme values
            mins = [entry['min'] for entry in filtered_data.values()]
            maxs = [entry['max'] for entry in filtered_data.values()]
            
            # Extreme values
            min_min = min(mins)
            min_max = max(mins)
            max_min = min(maxs)
            max_max = max(maxs)
            
            # Store data
            result = {
                'Group': group,
                'Classe': classe,
                'min_min': min_min,
                'min_max': min_max,
                'min_base': base_min,
                'max_min': max_min,
                'max_max': max_max,
                'max_base': base_max,
                'min_diff_low': base_min - min_min,
                'min_diff_high': min_max - base_min,
                'max_diff_low': base_max - max_min,
                'max_diff_high': max_max - base_max
            }
            results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Visualize by group
    visualize_dataframe_by_group(df, groups)
    
    # Visualize by class
    visualize_dataframe_by_class(df)
    
    return df

def visualize_dataframe_by_group(df, groups):
    # Min differences
    plt.figure(figsize=(15, 6))
    
    # Prepare data
    min_diff_low = [df[df['Group'] == group]['min_diff_low'].mean() for group in groups]
    min_diff_high = [df[df['Group'] == group]['min_diff_high'].mean() for group in groups]
    
    # Plot
    width = 0.35
    x = np.arange(len(groups))
    
    plt.bar(x - width/2, min_diff_low, width, label='Lowest Min Diff', color='lightblue')
    plt.bar(x + width/2, min_diff_high, width, label='Highest Min Diff', color='blue')
    
    plt.xlabel('Groups')
    plt.ylabel('Average Min Difference')
    plt.title('Average Min Differences by Group')
    plt.xticks(x, groups)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Dice_group_min_differences.png')
    plt.close()

    # Max differences
    plt.figure(figsize=(15, 6))
    
    # Prepare data
    max_diff_low = [df[df['Group'] == group]['max_diff_low'].mean() for group in groups]
    max_diff_high = [df[df['Group'] == group]['max_diff_high'].mean() for group in groups]
    
    # Plot
    plt.bar(x - width/2, max_diff_low, width, label='Lowest Max Diff', color='lightcoral')
    plt.bar(x + width/2, max_diff_high, width, label='Highest Max Diff', color='red')
    
    plt.xlabel('Groups')
    plt.ylabel('Average Max Difference')
    plt.title('Average Max Differences by Group')
    plt.xticks(x, groups)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Dice_group_max_differences.png')
    plt.close()

def visualize_dataframe_by_class(df):
    # Min differences by class
    plt.figure(figsize=(15, 6))
    
    classes = df['Classe'].unique()
    
    min_diff_low = [df[df['Classe'] == classe]['min_diff_low'].mean() for classe in classes]
    min_diff_high = [df[df['Classe'] == classe]['min_diff_high'].mean() for classe in classes]
    
    width = 0.35
    x = np.arange(len(classes))
    
    plt.bar(x - width/2, min_diff_low, width, label='Lowest Min Diff', color='lightblue')
    plt.bar(x + width/2, min_diff_high, width, label='Highest Min Diff', color='blue')
    
    plt.xlabel('Classes')
    plt.ylabel('Average Min Difference')
    plt.title('Average Min Differences by Class')
    plt.xticks(x, classes)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Dice_class_min_differences.png')
    plt.close()

    # Max differences by class
    plt.figure(figsize=(15, 6))
    
    max_diff_low = [df[df['Classe'] == classe]['max_diff_low'].mean() for classe in classes]
    max_diff_high = [df[df['Classe'] == classe]['max_diff_high'].mean() for classe in classes]
    
    plt.bar(x - width/2, max_diff_low, width, label='Lowest Max Diff', color='lightcoral')
    plt.bar(x + width/2, max_diff_high, width, label='Highest Max Diff', color='red')
    
    plt.xlabel('Classes')
    plt.ylabel('Average Max Difference')
    plt.title('Average Max Differences by Class')
    plt.xticks(x, classes)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Dice_class_max_differences.png')
    plt.close()

def main():
    # Filename of the JSON data
    filename = 'dice_min_max_results_group.json'
    
    # Create DataFrame
    df = create_statistics_dataframe(filename)
    
    # Print detailed statistics
    print(df.to_string(index=False))
    
    # Optional: Detailed summary statistics
    print("\nSummary Statistics:")
    print(df.describe())

if __name__ == '__main__':
    main()
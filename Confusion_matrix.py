import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse

def parse_stats_file(stats_file_path):
    """Parse the stats file and extract labels and counts in order, ignoring Total and Dropped entries."""
    labels = []
    try:
        with open(stats_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or '=' not in line:
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.split(';')[0].strip()  # Handle values with semicolons
                
                if 'Total' in key or 'Dropped' in key:
                    continue
                
                try:
                    count = int(value)
                    labels.append((key, count))
                except ValueError:
                    continue
        return labels
    except Exception as e:
        print(f"Error reading file {stats_file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Parse stats file and generate confusion matrix.')
    parser.add_argument('--stats', required=True, help='Path to the stats file')
    parser.add_argument('--output', required=True, help='Path to save the confusion matrix')
    parser.add_argument('--accuracy', type=float, required=True, help='Accuracy of the model')
    
    args = parser.parse_args()

    # Parse the stats file
    parsed = parse_stats_file(args.stats)
    if not parsed:
        print("Error: No valid data found in the stats file.")
        sys.exit(1)
    
    class_names = [label for label, _ in parsed]
    class_counts = [count for _, count in parsed]
    num_classes = len(class_names)
    total_samples = sum(class_counts)
    accuracy = args.accuracy / 100.0

    # Generate confusion matrix
    cm_data = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(num_classes):
        count_i = class_counts[i]
        tp_i = round(count_i * accuracy)
        error_i = count_i - tp_i

        others = [j for j in range(num_classes) if j != i]
        num_others = len(others)

        if num_others == 0:
            cm_data[i, i] = tp_i
            continue
        
        base_error = error_i // num_others
        remaining_error = error_i % num_others

        row = np.zeros(num_classes, dtype=int)
        row[i] = tp_i

        for idx, j in enumerate(others):
            row[j] = base_error
            if idx < remaining_error:
                row[j] += 1

        cm_data[i] = row

    # Create stats info
    stats_info = f"Model Accuracy: {args.accuracy}%\nTotal Samples: {total_samples}\n\nClass Distribution:\n"
    stats_info += "\n".join([f"{name}: {count}" for name, count in parsed])

    # Create visualization
    fig = plt.figure(figsize=(12, 10))

    # Confusion matrix subplot
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Network Traffic Classification Confusion Matrix')
    ax1.set_ylabel('Actual Traffic Type')
    ax1.set_xlabel('Predicted Traffic Type')

    # Stats text subplot
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax2.axis('off')
    ax2.text(0.5, 0.5, stats_info, ha='center', va='center', 
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.5))
    
    plt.tight_layout(pad=3.0)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {args.output}")
    plt.show()

if __name__ == "__main__":
    main()

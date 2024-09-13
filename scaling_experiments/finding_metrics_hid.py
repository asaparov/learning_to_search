import argparse
import os
import re
from collections import defaultdict
import logging
import numpy as np
import random
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def parse_seeds(seeds_arg):
    if ':' in seeds_arg:
        start, end = map(int, seeds_arg.split(':'))
        return range(start, end + 1)
    else:
        return [int(seed) for seed in seeds_arg.split(',')]

def extract_metric(file_path, epoch, metric):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Define regex patterns for each metric
    patterns = {
        'training_loss': r'epoch = {}, training loss = ([\d.]+)'.format(epoch),
        'training_accuracy': r'epoch = {}.*?training accuracy: ([\d.]+)'.format(epoch),
        'test_accuracy': r'epoch = {}.*?test accuracy = ([\d.]+)'.format(epoch),
        'test_loss': r'epoch = {}.*?test loss = ([\d.]+)'.format(epoch)
    }
    
    match = re.search(patterns[metric], content, re.DOTALL)
    if match:
        return float(match.group(1))
    else:
        return None

def process_directories(base_dir, epoch, metric, seeds):
    values = defaultdict(list)
    missing_data = defaultdict(list)
    missing_files = defaultdict(list)
    present_files = defaultdict(list)
    
    expected_layers = [16, 64, 256, 1024]  # Define the expected layers

    for seed in seeds:
        seed_dir = f"{base_dir}/seed_{seed}"
        if not os.path.exists(seed_dir):
            logger.warning(f"Directory not found: {seed_dir}")
            continue
        
        found_layers = set()
        
        for file in os.listdir(seed_dir):
            
            if file.startswith("scaling_n_c_1e-5_seed") and re.search(r"_hid_\d+\.out$", file):
                # print("for seed file", file)
                parts = file.split('_')
                layers = int(parts[-1].split('.')[0])
                found_layers.add(layers)
                present_files[layers].append(seed)
                
                file_path = os.path.join(seed_dir, file)
                value = extract_metric(file_path, epoch, metric)
                # print("\nvalue:", value)   
                
                if value is not None:
                    values[layers].append(value)
                else:
                    missing_data[layers].append(seed)
        
        # Check for missing layer files
        missing_layers = set(expected_layers) - found_layers
        for layer in missing_layers:
            missing_files[layer].append(seed)
    
    return values, missing_data, missing_files, present_files

def calculate_average_values(values):
    avg_values = {}
    min_values = {}
    max_values = {}
    for layer, value_list in values.items():
        if value_list:
            avg_values[layer] = sum(value_list) / len(value_list)
            min_values[layer] = min(value_list)
            max_values[layer] = max(value_list)
        else:
            logger.warning(f"No valid values found for layer {layer}")
    return avg_values, min_values, max_values

def main():
    parser = argparse.ArgumentParser(description="Calculate average metrics across seeds for different layers.")
    parser.add_argument("--epoch", type=int, default=1000, help="Epoch number to analyze (default: 1000)")
    parser.add_argument("--metric", choices=['training_loss', 'training_accuracy', 'test_accuracy', 'test_loss'], 
                        default='test_loss', help="Metric to average (default: test_loss)")
    parser.add_argument("--seeds", default="1:15", help="Seeds to process. Can be a range (e.g., '1:15') or a comma-separated list (e.g., '1,3,5,7,9')")
    
    args = parser.parse_args()

    ## Set your base directory to find seed folders ex. seed_1, seed_2, ...
    base_directory = "/scratch/sxp8182/work_with_abu/learning_to_search/cirriculum_learning/learning_to_search/"

    ## Define the hidden dims you want to analyze. Even though the variable name says layers, its actually hidden dims.
    layers = [16, 64, 256, 1024]  # Define the layers
    max_infinity = np.inf
    min_infinity = -np.inf

    # Add a random number to the output file name
    output_file = "average_metrics_hidden_" + str(random.randint(1, 1000000)) + ".txt"  # Specify the output file name
    print("saving in file:\n", output_file)
    with open(output_file, 'w') as f:  # Open the file for writing
        for epoch in tqdm(range(1, args.epoch + 1)):  # Loop through epochs from 1 to 900
            seeds = parse_seeds(args.seeds)
            values, missing_data, missing_files, present_files = process_directories(base_directory, epoch, args.metric, seeds)
            # print("value dict:\n\n")
            #print(values)
            avg_values, min_values, max_values = calculate_average_values(values)
            # print("avg_values:\n\n")
            #print(avg_values)
            # print("min_values:\n\n")
            #print(min_values)

            for missing_file in missing_files:
                print(f"Missing files for layer {missing_file}: {missing_files[missing_file]}")
            # for present_file in present_files:
            #     print(f"Present files for layer {present_file}: {present_files[present_file]}")
            for missing_data_no in missing_data:
                print(f"Missing data for layer {missing_data_no}: {missing_data[missing_data_no]}")

            # Store averages and mins in the specified format
            f.write(f"epoch_{epoch}_avg = [{', '.join(f'{avg_values.get(l, max_infinity):.6f}' for l in layers)}]\n")
            f.write(f"epoch_{epoch}_min = [{', '.join(f'{min_values.get(l, min_infinity):.6f}' for l in layers)}]\n")

if __name__ == "__main__":
    main()
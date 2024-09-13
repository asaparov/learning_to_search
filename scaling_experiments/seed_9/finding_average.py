import os
import re
from collections import defaultdict
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

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
        logger.warning(f"Epoch {epoch} {metric} not found in {file_path}")
        return None

def process_directories(base_dir, epoch, metric):
    values = defaultdict(list)
    missing_data = defaultdict(list)
    
    for seed in range(1, 10):  # seeds 1 to 9
        seed_dir = f"{base_dir}/seed_{seed}"
        if not os.path.exists(seed_dir):
            logger.warning(f"Directory not found: {seed_dir}")
            continue
        
        for file in os.listdir(seed_dir):
            if file.startswith("scaling_n_c_1e-5_") and file.endswith("_max_8.out"):
                parts = file.split('_')
                lookahead = int(parts[-3])
                
                file_path = os.path.join(seed_dir, file)
                value = extract_metric(file_path, epoch, metric)
                
                if value is not None:
                    values[lookahead].append(value)
                else:
                    missing_data[lookahead].append(seed)
    
    return values, missing_data

def calculate_average_values(values):
    avg_values = {}
    for lookahead, value_list in values.items():
        if value_list:
            avg_values[lookahead] = sum(value_list) / len(value_list)
        else:
            logger.warning(f"No valid values found for lookahead {lookahead}")
    return avg_values

def main():
    parser = argparse.ArgumentParser(description="Calculate average metrics across seeds for different lookaheads.")
    parser.add_argument("base_directory", help="Path to the base directory containing seed folders")
    parser.add_argument("--epoch", type=int, default=50, help="Epoch number to analyze (default: 1000)")
    parser.add_argument("--metric", choices=['training_loss', 'training_accuracy', 'test_accuracy', 'test_loss'], 
                        default='test_loss', help="Metric to average (default: test_loss)")
    
    args = parser.parse_args()

    values, missing_data = process_directories(args.base_directory, args.epoch, args.metric)
    avg_values = calculate_average_values(values)

    # Print results
    print(f"\nAverage {args.metric} for epoch {args.epoch}:")
    for lookahead in sorted(avg_values.keys()):
        print(f"Lookahead {lookahead}: {avg_values[lookahead]:.6f}")

    # Log missing data
    print("\nMissing data:")
    for lookahead, seeds in missing_data.items():
        logger.info(f"Lookahead {lookahead} missing data for seeds: {', '.join(map(str, seeds))}")

    # Log completely missing lookaheads
    all_lookaheads = set(range(32, 129, 6))
    found_lookaheads = set(avg_values.keys())
    missing_lookaheads = all_lookaheads - found_lookaheads
    if missing_lookaheads:
        logger.info(f"Completely missing lookaheads: {', '.join(map(str, sorted(missing_lookaheads)))}")

if __name__ == "__main__":
    main()
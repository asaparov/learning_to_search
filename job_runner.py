import os
import itertools
import shutil

def generate_sbatch_file(filename, params):
    with open(filename, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name={params['job_name']}
#SBATCH --output={params['output']}
#SBATCH --open-mode=append
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"

/scratch/sgp9467/learning_to_search/../pyenv/run_singularity.bash \\
    python /scratch/sgp9467/learning_to_search/train.py \\
    --max-input-size={params['max_input_size']} \\
    --dataset-size=-1 \\
    --max-lookahead={params['max_lookahead']} \\
    --seed={params['seed']} \\
    --nlayers={params['nlayers']} \\
    --hidden-dim=16 \\
    --bidirectional=y \\
    --absolute-pos-emb=y \\
    --learn-tok-emb=n \\
    --toeplitz-attn=n \\
    --toeplitz-reg=0.0 \\
    --toeplitz-pos-only=y \\
    --add-padding=y \\
    --ablate=none \\
    --preLN=y \\
    --curriculum=n \\
    --nl n \\
    --nl2 y \\
    --batch-size=128
""")

def main():
    experiments = [
        {'name': 'layers', 'param': 'nlayers', 'values': [8, 12, 16, 20, 24, 28, 32]},
        {'name': 'input_size', 'param': 'max_input_size', 'values': [32, 48, 64, 80, 96, 112, 128]},
        {'name': 'hidden_dim', 'param': 'hidden_dim', 'values': [16, 64, 256, 512]},
        # {'name': 'max_lookahead', 'param': 'max_lookahead', 'values': [4, 8, 12, 16, 20]}
        {"hidden_dim": 16, "batch_size": 128}
    ]

    seeds = [1, 2, 3]

    base_params = {
        'max_input_size': 64,
        'nlayers': 12,
        'seed': 1
    }

    if not os.path.exists('sbatch_files'):
        os.makedirs('sbatch_files')
    else: 
        shutil.rmtree('sbatch_files')
        os.makedirs('sbatch_files')
    for exp in experiments:
        for value, seed in itertools.product(exp['values'], seeds):
            params = base_params.copy()
            params[exp['param']] = value
            params['seed'] = seed
            
            job_name = f"{exp['name']}_{value}_seed{seed}"
            params['job_name'] = job_name
            params['output'] = f"run_{job_name}.out"
            
            filename = f"sbatch_files/job_{job_name}.sbatch"
            generate_sbatch_file(filename, params)
            print(f"Generated {filename}")

if __name__ == "__main__":
    main()

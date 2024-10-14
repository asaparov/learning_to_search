#!/bin/bash

for s in $(seq 1 5); do
	rm dfs_results/checkpoints_v3_3layer_inputsize48_maxlookahead10_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv 2> /dev/null
	rm dfs_results/checkpoints_v3_3layer_inputsize64_maxlookahead14_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv 2> /dev/null
	rm dfs_results/checkpoints_v3_3layer_inputsize80_maxlookahead18_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv 2> /dev/null
	rm dfs_results/checkpoints_v3_3layer_inputsize96_maxlookahead22_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv 2> /dev/null
	rm dfs_results/checkpoints_v3_3layer_inputsize112_maxlookahead26_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv 2> /dev/null
	rm dfs_results/checkpoints_v3_3layer_inputsize128_maxlookahead30_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv 2> /dev/null
	rm dfs_results/checkpoints_v3_3layer_inputsize144_maxlookahead34_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv 2> /dev/null
	rm dfs_results/checkpoints_v3_3layer_inputsize160_maxlookahead38_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv 2> /dev/null
	rm dfs_results/checkpoints_v3_3layer_inputsize176_maxlookahead42_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv 2> /dev/null
	rm dfs_results/checkpoints_v3_3layer_inputsize192_maxlookahead46_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv 2> /dev/null
done

for s in $(seq 1 5); do
	mkdir -p dfs_csvs/scaling_dfs_balanced/seed_${s}
done

for s in $(seq 208 267); do
	python output_to_json.py run_train${s}.out
done

for s in $(seq 1 5); do
	mv dfs_results/checkpoints_v3_3layer_inputsize48_maxlookahead10_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv dfs_csvs/scaling_dfs_balanced/seed_${s}/input_48.csv
	mv dfs_results/checkpoints_v3_3layer_inputsize64_maxlookahead14_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv dfs_csvs/scaling_dfs_balanced/seed_${s}/input_64.csv
	mv dfs_results/checkpoints_v3_3layer_inputsize80_maxlookahead18_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv dfs_csvs/scaling_dfs_balanced/seed_${s}/input_80.csv
	mv dfs_results/checkpoints_v3_3layer_inputsize96_maxlookahead22_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv dfs_csvs/scaling_dfs_balanced/seed_${s}/input_96.csv
	mv dfs_results/checkpoints_v3_3layer_inputsize112_maxlookahead26_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv dfs_csvs/scaling_dfs_balanced/seed_${s}/input_112.csv
	mv dfs_results/checkpoints_v3_3layer_inputsize128_maxlookahead30_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv dfs_csvs/scaling_dfs_balanced/seed_${s}/input_128.csv
	mv dfs_results/checkpoints_v3_3layer_inputsize144_maxlookahead34_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv dfs_csvs/scaling_dfs_balanced/seed_${s}/input_144.csv
	mv dfs_results/checkpoints_v3_3layer_inputsize160_maxlookahead38_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv dfs_csvs/scaling_dfs_balanced/seed_${s}/input_160.csv
	mv dfs_results/checkpoints_v3_3layer_inputsize176_maxlookahead42_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv dfs_csvs/scaling_dfs_balanced/seed_${s}/input_176.csv
	mv dfs_results/checkpoints_v3_3layer_inputsize192_maxlookahead46_seed${s}_trainstreaming_nomask_unablated_dfs/log.csv dfs_csvs/scaling_dfs_balanced/seed_${s}/input_192.csv
done

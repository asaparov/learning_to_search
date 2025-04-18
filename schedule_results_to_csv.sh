#!/bin/bash

for s in $(seq 1 5); do
	rm dfs_results/checkpoints_v3_16layer_inputsize112_maxlookahead26_seed${s}_trainstreaming_hiddendim128_nomask_unablated_dfs_nhead4/log.csv 2> /dev/null
	rm dfs_results/checkpoints_v3_16layer_inputsize112_maxlookahead26_seed${s}_trainstreaming_hiddendim128_nomask_unablated_dfs_nhead4_warmup262144/log.csv 2> /dev/null
	rm useful_path_results/checkpoints_v3_8layer_inputsize98_maxlookahead15_seed${s}_trainstreaming_nomask_unablated_padded/log.csv 2> /dev/null
	rm useful_path_results/checkpoints_v3_8layer_inputsize98_maxlookahead15_seed${s}_trainstreaming_nomask_unablated_padded_warmup262144/log.csv 2> /dev/null
done

for s in $(seq 1 5); do
	mkdir -p schedule_csvs/scaling_dfs_schedule/seed_${s}
	mkdir -p schedule_csvs/scaling_direct_schedule/seed_${s}
done

for s in $(seq 278 282); do
	python output_to_json.py run_train${s}.out
done
for s in $(seq 298 307); do
	python output_to_json.py run_train${s}.out
done

for s in $(seq 1 5); do
	mv dfs_results/checkpoints_v3_16layer_inputsize112_maxlookahead26_seed${s}_trainstreaming_hiddendim128_nomask_unablated_dfs_nhead4/log.csv schedule_csvs/scaling_dfs_schedule/seed_${s}/warmup_0.csv
	mv dfs_results/checkpoints_v3_16layer_inputsize112_maxlookahead26_seed${s}_trainstreaming_hiddendim128_nomask_unablated_dfs_nhead4_warmup262144/log.csv schedule_csvs/scaling_dfs_schedule/seed_${s}/warmup_262144.csv
	mv useful_path_results/checkpoints_v3_8layer_inputsize98_maxlookahead15_seed${s}_trainstreaming_nomask_unablated_padded/log.csv schedule_csvs/scaling_direct_schedule/seed_${s}/warmup_0.csv
	mv useful_path_results/checkpoints_v3_8layer_inputsize98_maxlookahead15_seed${s}_trainstreaming_nomask_unablated_padded_warmup262144/log.csv schedule_csvs/scaling_direct_schedule/seed_${s}/warmup_262144.csv
done

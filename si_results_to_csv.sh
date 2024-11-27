#!/bin/bash

for s in $(seq 1 1); do
	rm si_results/checkpoints_v3_4layer_inputsize80_maxlookahead0_seed${s}_trainstreaming_nomask_unablated_si_update65536/log.csv 2> /dev/null
	rm si_results/checkpoints_v3_4layer_inputsize112_maxlookahead0_seed${s}_trainstreaming_nomask_unablated_si_update65536/log.csv 2> /dev/null
	rm si_results/checkpoints_v3_4layer_inputsize144_maxlookahead0_seed${s}_trainstreaming_nomask_unablated_si_update65536/log.csv 2> /dev/null
	rm si_results/checkpoints_v3_4layer_inputsize176_maxlookahead0_seed${s}_trainstreaming_nomask_unablated_si_update65536/log.csv 2> /dev/null
	rm si_results/checkpoints_v3_4layer_inputsize272_maxlookahead0_seed${s}_trainstreaming_nomask_unablated_si_update65536/log.csv 2> /dev/null
done

for s in $(seq 1 1); do
	mkdir -p si_csvs/scaling_si/seed_${s}
done

for s in $(seq 2 5); do
	python output_to_json.py run_train${s}.out
done
python output_to_json.py run_train8.out

for s in $(seq 1 1); do
	mv si_results/checkpoints_v3_4layer_inputsize80_maxlookahead0_seed${s}_trainstreaming_nomask_unablated_si_update65536/log.csv si_csvs/scaling_si/seed_${s}/input_80.csv
	mv si_results/checkpoints_v3_4layer_inputsize112_maxlookahead0_seed${s}_trainstreaming_nomask_unablated_si_update65536/log.csv si_csvs/scaling_si/seed_${s}/input_112.csv
	mv si_results/checkpoints_v3_4layer_inputsize144_maxlookahead0_seed${s}_trainstreaming_nomask_unablated_si_update65536/log.csv si_csvs/scaling_si/seed_${s}/input_144.csv
	mv si_results/checkpoints_v3_4layer_inputsize176_maxlookahead0_seed${s}_trainstreaming_nomask_unablated_si_update65536/log.csv si_csvs/scaling_si/seed_${s}/input_176.csv
	mv si_results/checkpoints_v3_4layer_inputsize272_maxlookahead0_seed${s}_trainstreaming_nomask_unablated_si_update65536/log.csv si_csvs/scaling_si/seed_${s}/input_272.csv
done


for s in $(seq 1 2); do
	rm si_results/checkpoints_v3_4layer_inputsize272_maxlookahead0_seed${s}_trainstreaming_nomask_unablated_si_update65536/log.csv 2> /dev/null
	rm si_results/checkpoints_v3_4layer_inputsize272_maxlookahead0_seed${s}_trainstreaming_hiddendim128_nomask_unablated_si_update65536/log.csv 2> /dev/null
	rm si_results/checkpoints_v3_8layer_inputsize272_maxlookahead0_seed${s}_trainstreaming_hiddendim256_nomask_unablated_si_update65536/log.csv 2> /dev/null
	rm si_results/checkpoints_v3_4layer_inputsize272_maxlookahead0_seed${s}_trainstreaming_hiddendim512_nomask_unablated_si_update65536/log.csv 2> /dev/null
	rm si_results/checkpoints_v3_4layer_inputsize272_maxlookahead0_seed${s}_trainstreaming_hiddendim1024_nomask_unablated_si_update65536/log.csv 2> /dev/null
done

for s in $(seq 1 2); do
	mkdir -p si_csvs/scaling_si_params/seed_${s}
done

python output_to_json.py run_train8.out
for s in $(seq 19 27); do
	python output_to_json.py run_train${s}.out
done

for s in $(seq 1 2); do
	mv si_results/checkpoints_v3_4layer_inputsize272_maxlookahead0_seed${s}_trainstreaming_nomask_unablated_si_update65536/log.csv si_csvs/scaling_si_params/seed_${s}/4layer_16hid_1head.csv
	mv si_results/checkpoints_v3_4layer_inputsize272_maxlookahead0_seed${s}_trainstreaming_hiddendim128_nomask_unablated_si_update65536/log.csv si_csvs/scaling_si_params/seed_${s}/4layer_128hid_1head.csv
	mv si_results/checkpoints_v3_8layer_inputsize272_maxlookahead0_seed${s}_trainstreaming_hiddendim256_nomask_unablated_si_update65536/log.csv si_csvs/scaling_si_params/seed_${s}/8layer_256hid_1head.csv
	mv si_results/checkpoints_v3_4layer_inputsize272_maxlookahead0_seed${s}_trainstreaming_hiddendim512_nomask_unablated_si_update65536/log.csv si_csvs/scaling_si_params/seed_${s}/4layer_512hid_1head.csv
	mv si_results/checkpoints_v3_4layer_inputsize272_maxlookahead0_seed${s}_trainstreaming_hiddendim1024_nomask_unablated_si_update65536/log.csv si_csvs/scaling_si_params/seed_${s}/4layer_1024hid_1head.csv
done

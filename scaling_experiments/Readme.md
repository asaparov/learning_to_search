This folder is for scaling_experiments.
It contains results of all seeds from 1 to 14.

Each seed folder contains .sbatch, .err, and .out files for each experiments.

We have 3 files `finding_metrics_hid.py`, `finding_metrics_input_sizes.py` and `finding_metrics_layers.py` to analyze the experiments.

Steps to analyze the outputs of these experiments.

1) `finding_metrics_input_sizes.py`
- Set the `base_directory` in  the code.
- Run `python finding_metric_hid.py`, you can specify arguments `--epochs, --metrics, --seeds` you want to analyze.
- This code will calculate minimum and average across all the seeds for every epoch for the metric you have specified.

2) `finding_metrics_hid.py`
- Set the `base_directory` and `layers` you want to analyze in the code.
- Run `python finding_metric_hid.py`, you can specify arguments `--epochs, --metrics, --seeds` you want to analyze.
- This code will calculate minimum and average across all the seeds for every epoch for the metric you have specified.

3) `finding_metrics_hid.py`
- Set the `base_directory` and `layers` you want to analyze in the code.
- Run `python finding_metric_hid.py`, you can specify arguments `--epochs, --metrics, --seeds` you want to analyze.
- This code will calculate minimum and average across all the seeds for every epoch for the metric you have specified.


The outputs generated from these files can directly be given to the python notebooks for generating plots.
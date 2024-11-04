#!/usr/bin/python
import time

def build_module(name):
	from os import system
	if system(f"g++ -Ofast -fno-stack-protector -Wall -Wpedantic -shared -fPIC $(python3 -m pybind11 --includes) -I. {name}.cpp -o {name}$(python3-config --extension-suffix)") != 0:
		print(f"ERROR: Unable to compile `{name}.cpp`.")
		import sys
		sys.exit(1)
try:
	from os.path import getmtime
	from importlib.util import find_spec
	generator_module = find_spec('generator')
	if generator_module == None:
		raise ModuleNotFoundError
	elif getmtime(generator_module.origin) < getmtime('generator.cpp'):
		print("C++ module `generator` is out-of-date. Compiling from source...")
		build_module("generator")
	import generator
except ModuleNotFoundError:
	print("C++ module `generator` not found. Compiling from source...")
	build_module("generator")
	import generator
print("C++ module `generator` loaded.")

reserved_inputs = set()
start_time = time.perf_counter()
generator.set_seed(9)
dataset_size = 2 ** 20
#output = generator.generate_training_set(128, dataset_size, 2, 5, reserved_inputs, -1, False)
#output = generator.generate_dfs_training_set(48, dataset_size, reserved_inputs, 10, False, True, False)
output = generator.generate_si_training_set(48, dataset_size, reserved_inputs, 4, 7, True, False)
import pdb; pdb.set_trace()
print("Throughput: {} examples generated/s".format(dataset_size / (time.perf_counter() - start_time)))

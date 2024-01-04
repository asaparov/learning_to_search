#!/usr/bin/python
import time

def build_module(name):
	from os import system
	if system(f"g++ -Ofast -fno-stack-protector -Wall -Wpedantic -shared -fPIC $(python3 -m pybind11 --includes) -I../ {name}.cpp -o {name}$(python3-config --extension-suffix)") != 0:
		print(f"ERROR: Unable to compile `{name}.cpp`.")
		import sys
		sys.exit(1)
try:
	from os.path import getmtime
	from importlib.util import find_spec
	if getmtime(find_spec('generator').origin) < getmtime('generator.cpp'):
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
output = generator.generate_training_set(128, dataset_size, 10, reserved_inputs, -1, False)
print("Throughput: {} examples generated/s".format(dataset_size / (time.perf_counter() - start_time)))

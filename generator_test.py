#!/usr/bin/python
import time
try:
	import generator
except ModuleNotFoundError:
	print("C++ module `generator` not found. Compiling from source...")
	import os
	if os.system("g++ -Ofast -fno-stack-protector -Wall -Wpedantic -shared -fPIC $(python3 -m pybind11 --includes) -I../ generator.cpp -o generator$(python3-config --extension-suffix)") != 0:
		print("ERROR: Unable to compile `generator.cpp`.")
		import sys
		sys.exit(1)
	import generator
print("C++ module `generator` loaded.")

reserved_inputs = set()
start_time = time.perf_counter()
output = generator.generate_training_set(64, 2 ** 16, 6, reserved_inputs, -1, False)
print("Throughput: {} examples generated/s".format(2 ** 16 / (time.perf_counter() - start_time)))

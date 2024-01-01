#!/usr/bin/python
import time
import generator
reserved_inputs = set()
start_time = time.perf_counter()
output = generator.generate_training_set(64, 2 ** 16, 6, reserved_inputs, -1, False)
print("Throughput: {} examples generated/s".format(2 ** 16 / (time.perf_counter() - start_time)))

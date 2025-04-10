import os
import pickle
import sysconfig
import time
from os import listdir, makedirs, popen
from os.path import isfile, isdir
from random import sample, randrange, choice, shuffle, seed, getstate, setstate, Random
from sys import stdout

import numpy as np
from pybind11.__main__ import print_includes
from io import StringIO
import torch
from torch import nn, LongTensor
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader

from Sophia import SophiaG
from gpt2 import Transformer, TransformerLayer, ToeplitzMode, AblationMode, PositionEmbedding


def build_module(name):
    import sys
    old_stdout = sys.stdout
    try:
        sys.stdout = StringIO()
        print_includes()
        includes = sys.stdout.getvalue().strip()
        sys.stdout.close()
        sys.stdout = old_stdout
    except Exception as e:
        raise e
    finally:
        sys.stdout = old_stdout

    python_extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if sys.platform == "darwin":
        # macOS command
        command = (
            f"g++ -std=c++11 -Ofast -DNDEBUG -fno-stack-protector "
            f"-Wall -Wpedantic -undefined dynamic_lookup -shared -fPIC "
            f"{includes} -I. {name}.cpp -o {name}{python_extension_suffix}"
        )
    else:
        # Non-macOS command
        command = (
            f"g++ -Ofast -std=c++11 -DNDEBUG -fno-stack-protector "
            f"-Wall -Wpedantic -shared -fPIC "
            f"{includes} -I. {name}.cpp -o {name}{python_extension_suffix}"
        )
    print(command)
    if os.system(command) != 0:
        print(f"ERROR: Unable to compile `{name}.cpp`.")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    try:
        from os.path import getmtime
        from importlib.util import find_spec
        generator_spec = find_spec('generator')
        if generator_spec == None:
            raise ModuleNotFoundError
        if getmtime(generator_spec.origin) < getmtime('generator.cpp'):
            print("C++ module `generator` is out-of-date. Compiling from source...")
            build_module("generator")
        import generator
    except ModuleNotFoundError:
        print("C++ module `generator` not found. Compiling from source...")
        build_module("generator")
        import generator
    except ImportError:
        print("Error loading C++ module `generator`. Compiling from source...")
        build_module("generator")
        import generator
    print("C++ module `generator` loaded.")


def test_generate_si_training_set():
    max_input_size = 256
    max_edges = (max_input_size - 2) // 6
    max_frontier_size = (max_edges + 1) // 2
    max_branch_size = max_edges
    frontier_branches = []
    for frontier_size in range(1, max_frontier_size + 1):
        for branch_size in range(1, max_branch_size + 1):
            if frontier_size + branch_size > max_edges + 1:
                continue
            frontier_branches.append((frontier_size, branch_size))
    for frontier_size, branch_size in frontier_branches:

    # print(max_input_size, max_edges, max_frontier_size, max_branch_size)

        dataset_size = 1000
        reserved_inputs = set()
        uniform = False
        quiet = False
        alpha = 1.0

        # print(f"({frontier_size},{branch_size})")
        # 0 is default, 1 is selection only, 2 is inference
        result = generator.generate_si_training_set(
            max_input_size, dataset_size, reserved_inputs,
            frontier_size, branch_size,
            uniform, quiet, alpha, 2
        )

        inputs, outputs, labels, collisions = result

    # # Print some basic information about the generated arrays.
    #     print("Generated SI training samples (inference):")
    #     print("Inputs shape:", inputs.shape)
    #     print("Outputs shape:", outputs.shape)
    #     print("Labels shape:", labels.shape)
    #     print("Number of collisions:", collisions)
    #     print("\nSample inputs:")
    #     print(inputs)
    #     print("\nSample outputs:")
    #     print(outputs)
    #     print("\nLabels:")
    #     print(labels)


if __name__ == "__main__":
    test_generate_si_training_set()
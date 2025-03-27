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

def test_generate_si_training_set_split():
    max_input_size = 50
    dataset_size = 5
    reserved_inputs = set()
    requested_frontier_size = 2
    requested_branch_size = 2
    uniform = False
    quiet = True
    alpha = 1.0

    result = generator.generate_si_training_set_split(
        max_input_size, dataset_size, reserved_inputs,
        requested_frontier_size, requested_branch_size,
        uniform, quiet, alpha
    )

    inputs, outputs, sel_labels, inf_labels, collisions = result

    # Print some basic information about the generated arrays.
    print("Generated SI training samples (split version):")
    print("Inputs shape:", inputs.shape)
    print("Outputs shape:", outputs.shape)
    print("Selection labels shape:", sel_labels.shape)
    print("Inference labels shape:", inf_labels.shape)
    print("Number of collisions:", collisions)
    print("\nSample inputs:")
    print(inputs)
    print("\nSample outputs:")
    print(outputs)
    print("\nSelection labels (sel_labels):")
    print(sel_labels)
    print("\nInference labels (inf_labels):")
    print(inf_labels)


def test_generate_si_training_set():
    max_input_size = 50
    dataset_size = 5
    reserved_inputs = set()
    requested_frontier_size = 2
    requested_branch_size = 2
    uniform = False
    quiet = True
    alpha = 1.0

    # 0 is default, 1 is selection only, 2 is inference
    result = generator.generate_si_training_set(
        max_input_size, dataset_size, reserved_inputs,
        requested_frontier_size, requested_branch_size,
        uniform, quiet, alpha, 2
    )

    inputs, outputs, labels, collisions = result

    # Print some basic information about the generated arrays.
    print("Generated SI training samples (split version):")
    print("Inputs shape:", inputs.shape)
    print("Outputs shape:", outputs.shape)
    print("Labels shape:", labels.shape)
    print("Number of collisions:", collisions)
    print("\nSample inputs:")
    print(inputs)
    print("\nSample outputs:")
    print(outputs)
    print("\nLabels:")
    print(labels)


if __name__ == "__main__":
    test_generate_si_training_set()
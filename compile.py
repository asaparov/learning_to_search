import sys

def compile_if_needed():
    def build_module(name):
        from os import system
        if system(f"g++ -Ofast -fno-stack-protector -Wall -Wpedantic -shared -fPIC $(python3 -m pybind11 --includes) -I./ {name}.cpp -o {name}$(python3-config --extension-suffix)") != 0:
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

def compile_mapping_if_needed():
    def build_module(name):
        from os import system
        if system(f"g++ -Ofast -fno-stack-protector -Wall -Wpedantic -shared -fPIC $(python3 -m pybind11 --includes) -I./ {name}.cpp -o {name}$(python3-config --extension-suffix)") != 0:
            print(f"ERROR: Unable to compile `{name}.cpp`.")
            import sys
            sys.exit(1)
    try:
        from os.path import getmtime
        from importlib.util import find_spec
        mapping_module = find_spec('mapping')
        if mapping_module == None:
            raise ModuleNotFoundError
        elif getmtime(mapping_module.origin) < getmtime('mapping.cpp'):
            print("C++ module `mapping` is out-of-date. Compiling from source...")
            build_module("mapping")
        import mapping
    except ModuleNotFoundError:
        print("C++ module `mapping` not found. Compiling from source...")
        build_module("mapping")
        import mapping
    print("C++ module `mapping` loaded.")
    return mapping


if __name__ == "__main__":
    compile_mapping_if_needed()

    import mapping
        #   py::arg("data"), py::arg("output_tokens"), 
        #   py::arg("input_size"), py::arg("TRANSFORMER_LENGTH"), py::arg("verbose") = false,

    data = [[0,1,2,3],[1,2,3,4]]
    outout_tokens = [1,2]
    input_size = 32
    TRANASFORMER_LENGTH = 128
    verbose = False

    out = mapping.map_tokens_to_natural_language_batched(data, outout_tokens, input_size, TRANASFORMER_LENGTH, verbose)

    print(out)
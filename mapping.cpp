// map_tokens.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include<iostream>

namespace py = pybind11;

// Constants (these should be defined elsewhere and imported, but for this example we'll define them here)
std::vector<std::string> NAMES = {"Alex", "Bob", "Charlie", "David", "Eve", "Fred", "Gina", "Hank", "Ivy", "Jack", 
                                  "Kyle", "Lily", "Mia", "Nate", "Olivia", "Pam", "Quinn", "Ryan", "Sam", "Tara", 
                                  "Uma", "Victor", "Wendy", "Xavier", "Yara", "Zara"};

std::vector<std::string> NOUNS = {"qumpus", "shumpus", "grumpus", "plumpus", "clumpus", "kumpus", "sumpus", "slumpus", "umpus", 
                                  "flumpus", "lumpus", "rumpus", "numpus", "glumpus", "mumpus", "tumpus", "humpus", "bumpus", 
                                  "pumpus", "xumpus", "wumpus", "jumpus", "yumpus", "zumpus", "blumpus", "dumpus", "frumpus", "vumpus"};

std::unordered_map<std::string, std::string> CONNECTORS = {
    {"is a", "singular"},
    {"has", "plural"},
    {"wants a", "singular"},
    {"likes many", "plural"},
    {"cares for a", "singular"},
    {"is friends with", "plural"}
};

// Global variables
std::vector<std::string> ALL_ATOMS;

std::vector<std::string> generate_atoms(int atom_count) {
    std::unordered_set<std::string> atoms;
    std::random_device rd;
    std::mt19937 gen(rd());

    while (atoms.size() < static_cast<size_t>(atom_count)) {
        std::vector<std::string> connector_keys;
        for (const auto& pair : CONNECTORS) {
            connector_keys.push_back(pair.first);
        }
        std::uniform_int_distribution<> connector_dist(0, connector_keys.size() - 1);
        std::string connector = connector_keys[connector_dist(gen)];

        std::uniform_int_distribution<> noun_dist(0, NOUNS.size() - 1);
        std::string predicate = NOUNS[noun_dist(gen)];

        if (CONNECTORS[connector] == "plural") {
            predicate += "es";
        }

        std::uniform_int_distribution<> name_dist(0, NAMES.size() - 1);
        std::string name = NAMES[name_dist(gen)];

        atoms.insert(name + " " + connector + " " + predicate + " .");
    }

    return std::vector<std::string>(atoms.begin(), atoms.end());
}

std::string generate_edge(int src, int dest, const std::unordered_map<int, std::string>& atom_map) {
    return "If " + atom_map.at(src).substr(0, atom_map.at(src).length() - 1) + ", then " + atom_map.at(dest);
}

std::pair<std::vector<std::string>, std::vector<std::string>>
map_tokens_to_natural_language(const std::vector<int>& tokens, int output, int max_input_size, bool verbose = false) {
    int QUERY_PREFIX_TOKEN = (max_input_size - 5) / 3 + 4;
    int PADDING_TOKEN = (max_input_size - 5) / 3 + 3;
    int EDGE_PREFIX_TOKEN = (max_input_size - 5) / 3 + 2;
    int PATH_PREFIX_TOKEN = (max_input_size - 5) / 3 + 1;

    std::vector<std::string> unique_atoms;
    std::unordered_set<int> unique_token_set;
    for (const auto& token : tokens) {
        if (token != QUERY_PREFIX_TOKEN && token != PADDING_TOKEN && token != EDGE_PREFIX_TOKEN && token != PATH_PREFIX_TOKEN) {
            unique_token_set.insert(token);
        }
    }
    size_t num_unique_tokens = unique_token_set.size();

    // Sample num_unique_tokens from ALL_ATOMS array
    std::vector<int> indices(ALL_ATOMS.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    for (size_t i = 0; i < num_unique_tokens && i < indices.size(); ++i) {
        unique_atoms.push_back(ALL_ATOMS[indices[i]]);
    }

    std::vector<int> unique_tokens(unique_token_set.begin(), unique_token_set.end());

    std::unordered_map<int, std::string> token_to_atom;
    for (size_t i = 0; i < unique_tokens.size(); ++i) {
        token_to_atom[unique_tokens[i]] = unique_atoms[i];
    }

    std::vector<std::string> out_tokens;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == QUERY_PREFIX_TOKEN) {
            out_tokens.push_back("Given");
            out_tokens.push_back(token_to_atom[tokens[i+1]].substr(0, token_to_atom[tokens[i+1]].length() - 1) + ",");
            out_tokens.push_back("prove");
            out_tokens.push_back(token_to_atom[tokens[i+2]]);
            i += 2;
        } else if (tokens[i] == EDGE_PREFIX_TOKEN) {
            out_tokens.push_back(generate_edge(tokens[i+1], tokens[i+2], token_to_atom));
            i += 2;
        } else if (tokens[i] == PATH_PREFIX_TOKEN) {
            break;
        }
    }

    if (verbose) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << tokens[i] << " -> " << (i < out_tokens.size() ? out_tokens[i] : "N/A") << std::endl;
        }
    }

    std::string full_out = "";
    for (const auto& token : out_tokens) {
        full_out += token + " ";
    }
    full_out = full_out.substr(0, full_out.length() - 1);  // Remove trailing space

    std::istringstream iss(token_to_atom[output]);
    std::vector<std::string> atoms{std::istream_iterator<std::string>{iss},
                                   std::istream_iterator<std::string>{}};

    std::vector<std::string> examples = {full_out};
    std::vector<std::string> labels = {atoms[0]};

    for (size_t i = 0; i < atoms.size() - 1; ++i) {
        full_out += " " + atoms[i];
        examples.push_back(full_out);
        labels.push_back(atoms[i+1]);
    }

    return {examples, labels};
}

std::tuple<std::vector<std::string>, std::vector<std::string>> 
map_tokens_to_natural_language_batched(
    const std::vector<std::vector<int>>& data,
    const std::vector<int>& output_tokens,
    int input_size,
    int TRANSFORMER_LENGTH,
    bool verbose = false
) {
    std::vector<std::string> all_tok;
    std::vector<std::string> all_out;

    for (size_t i = 0; i < data.size(); ++i) {
        const auto& tokens = data[i];
        int output = output_tokens[i];

        std::vector<std::string> examples, labels;
        std::tie(examples, labels) = map_tokens_to_natural_language(tokens, output, input_size, verbose);

        all_tok.insert(all_tok.end(), examples.begin(), examples.end());
        all_out.insert(all_out.end(), labels.begin(), labels.end());
    }

    return std::make_tuple(all_tok, all_out);
}

PYBIND11_MODULE(mapping, m) {
    m.def("map_tokens_to_natural_language_batched", &map_tokens_to_natural_language_batched, 
          py::arg("data"), py::arg("output_tokens"), 
          py::arg("input_size"), py::arg("TRANSFORMER_LENGTH"), py::arg("verbose") = false,
          "Map tokens to natural language in batches");

    m.def("generate_atoms", &generate_atoms, py::arg("atom_count"),
          "Generate a list of atoms");

    // Initialize ALL_ATOMS
    ALL_ATOMS = generate_atoms(1000);
}
// mapping.cpp
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
#include <iostream>
#include <chrono>
#include <numeric>

namespace py = pybind11;

// Constants
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

std::vector<std::string> VOCAB;
std::unordered_map<std::string, int> VOCAB_DICT;

// Global variables
std::vector<std::vector<int>> ALL_ATOMS;

void initialize_vocab() {
    VOCAB = NAMES;
    VOCAB.insert(VOCAB.end(), NOUNS.begin(), NOUNS.end());
    for (const auto& noun : NOUNS) {
        VOCAB.push_back(noun + "es");
    }
    std::vector<std::string> additional_words = {"a", "is", "has", "wants", "likes", "cares", "for", "friends", "with", "then", "Given", "If", "prove", "many", ".", " ", ",", "\n", ":", "Query", "Prefix", "Statements", "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"};
    VOCAB.insert(VOCAB.end(), additional_words.begin(), additional_words.end());

    for (size_t i = 0; i < VOCAB.size(); ++i) {
        VOCAB_DICT[VOCAB[i]] = i;
    }
}

std::vector<std::vector<int>> generate_atoms(int atom_count) {
    std::unordered_map<std::string, std::vector<int>> atoms;
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

        std::vector<int> atom = {VOCAB_DICT[name]};
        std::istringstream iss(connector);
        std::string word;
        while (iss >> word) {
            atom.push_back(VOCAB_DICT[word]);
        }
        atom.push_back(VOCAB_DICT[predicate]);

        std::string atom_key;
        for (int token : atom) {
            atom_key += std::to_string(token) + " ";
        }
        atoms[atom_key] = atom;
    }

    std::vector<std::vector<int>> result;
    for (const auto& pair : atoms) {
        result.push_back(pair.second);
    }
    return result;
}

std::vector<int> generate_edge(int src, int dest, const std::unordered_map<int, std::vector<int>>& atom_map) {
    std::vector<int> edge = {VOCAB_DICT["If"]};
    edge.insert(edge.end(), atom_map.at(src).begin(), atom_map.at(src).end());
    edge.push_back(VOCAB_DICT[","]);
    edge.push_back(VOCAB_DICT["then"]);
    edge.insert(edge.end(), atom_map.at(dest).begin(), atom_map.at(dest).end());
    return edge;
}

std::pair<std::vector<std::vector<int>>, std::vector<int>>
map_tokens_to_natural_language(const std::vector<int>& tokens, int output, int max_input_size, int TRANSFORMER_LENGTH, bool verbose = false) {
    std::vector<int> special_tokens = {
        (max_input_size - 5) / 3 + 4,
        (max_input_size - 5) / 3 + 3,
        (max_input_size - 5) / 3 + 2,
        (max_input_size - 5) / 3 + 1
    };
    int QUERY_PREFIX_TOKEN = special_tokens[0];
    int EDGE_PREFIX_TOKEN = special_tokens[2];
    int PATH_PREFIX_TOKEN = special_tokens[3];

    std::unordered_set<int> unique_tokens;
    for (int token : tokens) {
        if (std::find(special_tokens.begin(), special_tokens.end(), token) == special_tokens.end()) {
            unique_tokens.insert(token);
        }
    }

    std::vector<int> indices(ALL_ATOMS.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    std::unordered_map<int, std::vector<int>> token_to_atom;
    auto it = unique_tokens.begin();
    for (size_t i = 0; i < unique_tokens.size() && i < indices.size(); ++i, ++it) {
        token_to_atom[*it] = ALL_ATOMS[indices[i]];
    }

    std::vector<int> out_tokens;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == QUERY_PREFIX_TOKEN) {
            out_tokens.push_back(VOCAB_DICT["Given"]);
            out_tokens.insert(out_tokens.end(), token_to_atom[tokens[i+1]].begin(), token_to_atom[tokens[i+1]].end());
            out_tokens.push_back(VOCAB_DICT[","]);
            out_tokens.push_back(VOCAB_DICT["prove"]);
            out_tokens.insert(out_tokens.end(), token_to_atom[tokens[i+2]].begin(), token_to_atom[tokens[i+2]].end());
            out_tokens.push_back(VOCAB_DICT["."]);
            i += 2;
        } else if (tokens[i] == EDGE_PREFIX_TOKEN) {
            std::vector<int> edge = generate_edge(tokens[i+1], tokens[i+2], token_to_atom);
            out_tokens.insert(out_tokens.end(), edge.begin(), edge.end());
            out_tokens.push_back(VOCAB_DICT["."]);
            i += 2;
        } else if (tokens[i] == PATH_PREFIX_TOKEN) {
            break;
        }
    }

    if (verbose) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::cout << tokens[i] << " -> " << (i < out_tokens.size() ? out_tokens[i] : -1) << std::endl;
        }
    }

    std::vector<int> atoms = token_to_atom[output];
    atoms.push_back(VOCAB_DICT["."]);


    std::vector<std::vector<int>> examples = {out_tokens};
    for (size_t i = 0; i < atoms.size() - 1; ++i) {
        out_tokens.push_back(atoms[i]);
        examples.push_back(out_tokens);
    }

    return {examples, atoms};
}

std::tuple<py::array_t<int>, py::array_t<float>, py::array_t<long>, py::array_t<int>> 
map_tokens_to_natural_language_batched(
    const std::vector<std::vector<int>>& data,
    const std::vector<int>& output_tokens,
    int input_size,
    int TRANSFORMER_LENGTH,
    bool verbose = false
) {

    std::vector<std::vector<int>> all_tok;
    std::vector<int> all_out;
    std::vector<int> lengths;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < data.size(); ++i) {
        const auto& tokens = data[i];
        int output = output_tokens[i];

        std::vector<std::vector<int>> examples;
        std::vector<int> labels;
        std::tie(examples, labels) = map_tokens_to_natural_language(tokens, output, input_size, TRANSFORMER_LENGTH, verbose);


        all_tok.insert(all_tok.end(), examples.begin(), examples.end());
        all_out.insert(all_out.end(), labels.begin(), labels.end());
        lengths.push_back(examples.size());
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;

    start_time = std::chrono::high_resolution_clock::now();
    
    // Create the padded array
    py::array_t<int> padded_array({static_cast<py::ssize_t>(all_tok.size()), static_cast<py::ssize_t>(TRANSFORMER_LENGTH)});
    auto padded_array_unchecked = padded_array.mutable_unchecked<2>();
    
    for (py::ssize_t i = 0; i < padded_array_unchecked.shape(0); ++i) {
        py::ssize_t offset = static_cast<py::ssize_t>(TRANSFORMER_LENGTH - all_tok[i].size());
        for (py::ssize_t j = 0; j < padded_array_unchecked.shape(1); ++j) {
            if (j < offset) {
                padded_array_unchecked(i, j) = VOCAB_DICT["[PAD]"];
            } else {
                padded_array_unchecked(i, j) = all_tok[i][j - offset];
            }
        }
    }
    
    end_time = std::chrono::high_resolution_clock::now();
    diff = end_time - start_time;

    py::array_t<long> all_out_array = py::cast(all_out);
    py::array_t<int> lengths_array = py::cast(lengths);

    py::array_t<float> all_out_vec({static_cast<py::ssize_t>(all_out.size()), static_cast<py::ssize_t>(VOCAB.size())});
    auto all_out_vec_unchecked = all_out_vec.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < all_out_vec_unchecked.shape(0); ++i) {
        for (py::ssize_t j = 0; j < all_out_vec_unchecked.shape(1); ++j) {
            all_out_vec_unchecked(i, j) = (j == all_out[i]) ? 1.0f : 0.0f;
        }
    }

    return std::make_tuple(padded_array, all_out_vec, all_out_array, lengths_array);
}

PYBIND11_MODULE(mapping, m) {
    m.def("map_tokens_to_natural_language_batched", &map_tokens_to_natural_language_batched, 
          py::arg("data"), py::arg("output_tokens"), 
          py::arg("input_size"), py::arg("TRANSFORMER_LENGTH"), py::arg("verbose") = false,
          "Map tokens to natural language in batches");

    m.def("generate_atoms", &generate_atoms, py::arg("atom_count"),
          "Generate a list of atoms");

    // Initialize VOCAB and VOCAB_DICT
    initialize_vocab();

    // Initialize ALL_ATOMS
    ALL_ATOMS = generate_atoms(1000);
}
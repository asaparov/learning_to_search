#!/usr/bin/python

import json
import numpy as np
import time
import random
import torch
import os
from compile import compile_if_needed
from transformers import PreTrainedTokenizerFast
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast

compile_if_needed()
import generator

from vocab import NAMES, NOUNS, CONNECTORS, VOCAB
# from test_tk import tokenizer

def generate_atoms(atom_count):
    atoms = set()
    while len(atoms)<atom_count:
        connector = random.choice(list(CONNECTORS.keys()))
        predicate = random.choice(NOUNS)
        if CONNECTORS[connector] == 'plural':
            predicate += 'es'
        atoms.add(f"{random.choice(NAMES)} {connector} {predicate}.")
	# create a map from indices to atoms
    # atom_map = {i:atom for i, atom in enumerate(atoms)}
    atoms = list(atoms)
    random.shuffle(atoms)
    return atoms


def generate_edge(src, dest, atom_map):
    return [f"If {atom_map[src][:-1]},", f"then {atom_map[dest][:-1]}."]

def map_tokens_to_natural_language(tokens, output, max_input_size, verbose=False):

    QUERY_PREFIX_TOKEN = (max_input_size - 5) // 3 + 4
    PADDING_TOKEN = (max_input_size - 5) // 3 + 3
    EDGE_PREFIX_TOKEN = (max_input_size - 5) // 3 + 2
    PATH_PREFIX_TOKEN = (max_input_size - 5) // 3 + 1

    # count distinct numbers in the array
    unique_tokens = np.unique(tokens)
    # Remove special tokens from the unique_tokens array
    unique_tokens = unique_tokens[~np.isin(unique_tokens, [QUERY_PREFIX_TOKEN, PADDING_TOKEN, EDGE_PREFIX_TOKEN, PATH_PREFIX_TOKEN])]

    # now generate that number of atoms and predicates
    atoms = generate_atoms(len(unique_tokens))

    # create a mapping from tokens to atoms
    token_to_atom = {token: atoms[i] for i, token in enumerate(unique_tokens)}

    # translate the tokens to natural language
    out_tokens = []
    i = 0
    try:
        # out_tokens.append(["Statements:"])
        while i < len(tokens):
            if tokens[i] == QUERY_PREFIX_TOKEN:
                # out_tokens.append(["\nQuery:"])
                out_tokens.append('Given')
                out_tokens.append(token_to_atom[tokens[i+1]][:-1] + ',')
                out_tokens.append('prove')
                out_tokens.append(token_to_atom[tokens[i+2]])
                i+=2
            elif tokens[i] == EDGE_PREFIX_TOKEN:
                # out_tokens.append(["<|edge_prefix|>"])
                edge = generate_edge(tokens[i+1], tokens[i+2], token_to_atom)
                out_tokens.extend(edge)
                i += 2
            elif tokens[i] == PATH_PREFIX_TOKEN:
                # out_tokens.append(["\nPrefix:"])
                while(i+1<len(tokens)):
                    if tokens[i+1] == PATH_PREFIX_TOKEN:
                        # out_tokens.append(["\nPath:"])
                        pass
                    else:
                        atom = token_to_atom[tokens[i+1]]
                        # out_tokens.append([atom])
                    i+=1
            elif tokens[i] == PADDING_TOKEN:
                # out_tokens.append(["<|padding|>"])
                pass
            i += 1
    except Exception as e:
        import traceback
        print("Stack trace:")
        print(traceback.format_exc())
        print(f"Error mapping tokens to natural language: {e}")
        import pdb; pdb.set_trace()

    if verbose:
        for i in range(len(tokens)):
            print(f"{tokens[i]} -> {out_tokens[i]}")

    # flattened_out_tokens = [item for sublist in out_tokens for item in sublist]
    full_out = " ".join(out_tokens) + "\n" + token_to_atom[int(output)]
    return full_out, token_to_atom[int(output)]

def map_tokens_to_natural_language_batched(tokenizer, data, output_tokens, input_size, TRANSFORMER_LENGTH, verbose=False):

    batch_size = data.shape[0]
    all_tok = []
    all_out = []
    
    for i in range(batch_size):
        tokens, output = map_tokens_to_natural_language(data[i], output_tokens[i], input_size, verbose)
        all_tok.append(tokens)
        all_out.append(output)
    
    # tokenize
    all_tok = tokenizer.batch_encode_plus(all_tok, return_tensors='pt', padding='max_length', pad_to_max_length=True, max_length=TRANSFORMER_LENGTH)['input_ids']
    all_out = tokenizer.batch_encode_plus(all_out, return_tensors='pt', padding='max_length', pad_to_max_length=True, max_length=TRANSFORMER_LENGTH)['input_ids']

    return all_tok, all_out

def tokenize_and_join(tokens):
    return ' '.join(tokens)



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Generate training set using DFS")
    parser.add_argument("--seed", type=int, default=9, help="Random seed for generator")
    parser.add_argument("--dataset_size", type=int, default=3, help="Size of the dataset to generate")
    parser.add_argument("--input_size", type=int, default=64, help="Size of input for each example")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument('--dfs', action='store_true', help='Use DFS')
    parser.add_argument('--max_lookahead', type=int, default=-1, help='Maximum lookahead distance')
    parser.add_argument('--requested_backtrack', type=int, default=4, help='Requested backtrack distance')
    args = parser.parse_args()

    generator.set_seed(args.seed)
    dataset_size = args.dataset_size
    reserved_inputs = set()

    print("generating dataset...")

    if args.dfs:
        generated_data = generator.generate_dfs_training_set(args.input_size, dataset_size, reserved_inputs, args.requested_backtrack, args.verbose)
    else:
        generated_data = generator.generate_training_set(args.input_size, dataset_size, 3, 7, reserved_inputs, -1, False)

    # for i, (data, output_tokens) in enumerate(zip(generated_data[0], generated_data[2])):
        # print(f"Example {i}:")
    tokens, output = map_tokens_to_natural_language_batched(tokenizer, generated_data[0], generated_data[2], args.input_size)
    import pdb; pdb.set_trace()
        # map
        # print("Input tokens:")
        # for a,b in zip(data, tokens):
        #     print(f"{a} -> {b}")

        # print("Output tokens:")
        # for a,b in zip([output_tokens], [output]):
        #     print(f"{a} -> {b}")

        # print("==================")

        # print(tokens)
        # print(' '.join(tokens))
        # print(output)
        # print('==================')
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


from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from tokenizers import processors
from tokenizers.pre_tokenizers import BertPreTokenizer


def create_custom_tokenizer(vocab, max_length, save_path="./"):
    tokenizer = Tokenizer(WordLevel(vocab=dict(zip(vocab, range(len(vocab)))), unk_token="[UNK]"))
    tokenizer.pre_tokenizer = BertPreTokenizer()

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        padding_side="left",
        max_length=max_length
    )

def generate_atoms(atom_count):
    atoms = set()
    while len(atoms) < atom_count:
        connector = random.choice(list(CONNECTORS.keys()))
        predicate = random.choice(NOUNS)
        if CONNECTORS[connector] == 'plural':
            predicate += 'es'
        atoms.add(f"{random.choice(NAMES)} {connector} {predicate} .")
    return list(atoms)

ALL_ATOMS = generate_atoms(1000)

def generate_edge(src, dest, atom_map):
    return f"If {atom_map[src][:-1]}, then {atom_map[dest]}"

def map_tokens_to_natural_language(tokens, output, max_input_size, verbose=False):
    QUERY_PREFIX_TOKEN, PADDING_TOKEN, EDGE_PREFIX_TOKEN, PATH_PREFIX_TOKEN = [
        (max_input_size - 5) // 3 + i for i in range(4, 0, -1)
    ]

    unique_tokens = np.unique(tokens)
    unique_tokens = unique_tokens[~np.isin(unique_tokens, [QUERY_PREFIX_TOKEN, PADDING_TOKEN, EDGE_PREFIX_TOKEN, PATH_PREFIX_TOKEN])]

    unique_atoms = random.sample(ALL_ATOMS, len(unique_tokens))
    token_to_atom = dict(zip(unique_tokens, unique_atoms))

    out_tokens = []
    i = 0
    while i < len(tokens):
        if tokens[i] == QUERY_PREFIX_TOKEN:
            out_tokens.extend(['Given', f"{token_to_atom[tokens[i+1]][:-1]},", 'prove', token_to_atom[tokens[i+2]]])
            i += 2
        elif tokens[i] == EDGE_PREFIX_TOKEN:
            out_tokens.append(generate_edge(tokens[i+1], tokens[i+2], token_to_atom))
            i += 2
        elif tokens[i] == PATH_PREFIX_TOKEN:
            i = len(tokens) - 1  # Skip to the end
        i += 1

    if verbose:
        for t, o in zip(tokens, out_tokens):
            print(f"{t} -> {o}")

    full_out = " ".join(out_tokens)
    atoms = token_to_atom[int(output)].split()
    
    examples = [full_out]
    labels = [atoms[0]]
    for i in range(len(atoms) - 1):
        full_out += f" {atoms[i]}"
        examples.append(full_out.strip())
        labels.append(atoms[i+1])
    return examples, labels

def map_tokens_to_natural_language_batched(tokenizer, data, output_tokens, input_size, TRANSFORMER_LENGTH, verbose=False):
    all_tok = []
    all_out = []
    
    for tokens, output in zip(data, output_tokens):
        examples, labels = map_tokens_to_natural_language(tokens, output, input_size, verbose)
        all_tok.extend(examples)
        all_out.extend(labels)
    
    all_tok = tokenizer.batch_encode_plus(all_tok, return_tensors='pt', padding='max_length', pad_to_max_length=True, max_length=TRANSFORMER_LENGTH)['input_ids']
    all_out = tokenizer.batch_encode_plus(all_out, return_tensors='pt')['input_ids'].squeeze()

    all_out_vec = torch.nn.functional.one_hot(all_out, num_classes=len(tokenizer.vocab)).float()

    return all_tok, all_out_vec, all_out


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
        generated_data = generator.generate_training_set(args.input_size, dataset_size, 3, 7, reserved_inputs, -1, False, False)

# py::tuple generate_training_set(const unsigned int max_input_size, const uint64_t dataset_size, const unsigned int max_lookahead, const unsigned int max_edges, const py::object& reserved_inputs, const int distance_from_start, const bool nl, const bool quiet=false)

    # for i, (data, output_tokens) in enumerate(zip(generated_data[0], generated_data[2])):
        # print(f"Example {i}:")

    tokenizer = create_custom_tokenizer(VOCAB, args.input_size)
    
    tokens, all_out_vec, output = map_tokens_to_natural_language_batched(tokenizer, generated_data[0], generated_data[2], args.input_size, 6 * args.input_size, args.verbose)
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
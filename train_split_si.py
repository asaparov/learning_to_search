import multiprocessing
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

RESERVED_INDICES = (0,)

class Node(object):
    def __init__(self, id):
        self.id = id
        self.children = []
        self.parents = []

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return 'n(' + str(self.id) + ')'

    def __repr__(self):
        return 'n(' + str(self.id) + ')'

def binomial_confidence_int(p, n):
    return 1.96 * np.sqrt(p * (1.0 - p) / n)

def evaluate_model(model, inputs, outputs):
    device = next(model.parameters()).device
    inputs = torch.tensor(inputs)
    outputs = torch.tensor(outputs)
    inputs = inputs.to(device)
    outputs = outputs.to(device)
    max_input_size = inputs.shape[1]

    if outputs.dim() == 2:
        loss_func = BCEWithLogitsLoss(reduction='mean')
    else:
        loss_func = CrossEntropyLoss(reduction='mean')
    logits, _ = model(inputs)
    loss = loss_func(logits[:, -1, :], outputs).item()

    predictions = torch.argmax(logits[:, -1, :], 1)
    # print("Predictions: ", predictions)
    if outputs.dim() == 2:
        acc = torch.sum(torch.gather(outputs, 1, torch.argmax(logits[:,-1,:],dim=1).unsqueeze(1))).item() / outputs.size(0)
    else:
        acc = sum(predictions == outputs).item() / len(predictions)
    return acc, loss, predictions

# New Dummy Dataset class for the split si models
class DummyDatasetSplit(Dataset):
    def __init__(self, inputs, outputs, inf_labels, device, x_type=LongTensor, y_type=LongTensor):
        self.x_data = x_type(inputs).to(device)
        self.out_data = outputs
        # self.sel_data = y_type(sel_labels).to(device)
        self.inf_data = y_type(inf_labels).to(device)
    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, idx):
        # Return a tuple of (input, output, inf_label)
        return (self.x_data[idx], self.out_data[idx], self.inf_data[idx])

def unique(x):
    y = []
    for e in x:
        if e not in y:
            y.append(e)
    return y

#############################
# PPO Helper Functions
#############################
# PPO hyperparameters
PPO_EPSILON = 0.2
PPO_EPOCHS = 4
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_COEF = 0.01
PPO_MINIBATCH_SIZE = 16


def get_policy_and_value(model, inputs):
    # Get the output from the transformer.
    x = model(inputs)
    # In evaluation mode, model(inputs) may return (x, past) so support that:
    if isinstance(x, tuple):
        logits, _ = x
    else:
        logits = x
    # Assume logits has shape [batch, seq_len, hidden_dim].
    # We use the last time step's hidden state.
    last_hidden = logits[:, -1, :]
    # Compute the value estimate using the value head.
    value = model.value_head(last_hidden)  # shape: [batch, 1]
    return logits, value

def ppo_update(model, optimizer, inputs, actions, old_log_probs, rewards, old_values):
    logits, values = get_policy_and_value(model, inputs)
    prob_dist = torch.softmax(logits, dim=1)
    new_log_probs = torch.log(torch.gather(prob_dist, 1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
    ratio = torch.exp(new_log_probs - old_log_probs)
    advantages = rewards - old_values.squeeze(1)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * advantages
    policy_loss = -torch.mean(torch.min(surr1, surr2))
    value_loss = torch.mean((values.squeeze(1) - rewards) ** 2)
    entropy = -torch.mean(torch.sum(prob_dist * torch.log(prob_dist + 1e-8), dim=1))
    total_loss = policy_loss + PPO_VALUE_COEF * value_loss - PPO_ENTROPY_COEF * entropy
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss.item(), policy_loss.item(), value_loss.item(), entropy.item()

def train(max_input_size, dataset_size, distribution, max_lookahead, seed_value, nlayers, nhead, hidden_dim, bidirectional, pos_emb, learnable_token_emb, toeplitz_attn, toeplitz_reg, toeplitz_pos_only, add_padding, ablate, pre_ln, curriculum_mode, looped, task, warm_up, batch_size, learning_rate, update_rate, grad_accumulation_steps, split_si, reinforce, loss):
    generator.set_seed(seed_value)
    seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    PADDING_TOKEN = (max_input_size-5) // 3 + 3
    BATCH_SIZE = batch_size // grad_accumulation_steps
    print('Number of available CPUs: {}'.format(os.cpu_count()))
    stdout.flush()

    if curriculum_mode == 'y':
        print("Using curriculum learning")

    if reinforce == 'y':
        print("Using PPO reinforcement learning")

    if loss == "bce":
        print("Using BCE loss")

    if curriculum_mode != 'n' and dataset_size != -1:
        print('ERROR: Curriculum learning is only supported with streaming training (i.e. dataset_size = -1).')
        stdout.flush()
        return
    if distribution in ("crafted", "crafted_no_prefix", "star") and max_lookahead == None:
        print('ERROR: Crafted or star training distribution is selected but `max_lookahead` argument is missing.')
        stdout.flush()
        return
    if distribution == "simple" and max_lookahead != None:
        print('ERROR: `max_lookahead` is not supported with the simple training distribution.')
        stdout.flush()
        return
    if distribution in ("crafted_no_prefix", "star") and task != "search":
        print('ERROR: Distributions `crafted_no_prefix` and `star` are only supported with task `search`.')
        stdout.flush()
        return
    if split_si == 'y' and task != "si":
        print('ERROR: `split_si` is only supported with task `si`.')
        stdout.flush()
        return
    if max_lookahead == None:
        max_lookahead = -1


    # first reserve some data for OOD testing
    random_state = getstate()
    np_random_state = np.random.get_state()
    torch_random_state = torch.get_rng_state()

    reserved_inputs = set()
    NUM_TEST_SAMPLES = 10000
    if task == 'si':
        NUM_TEST_SAMPLES = 500
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
            gen_eval_start_time = time.perf_counter()
            setstate(random_state)
            np.random.set_state(np_random_state)
            torch.set_rng_state(torch_random_state)

            # print('Reserving OOD test data (selection) for frontier_size = {}, branch_size = {}'.format(frontier_size, branch_size))
            # stdout.flush()
            # sel_inputs, sel_outputs, sel_labels ,_ = generator.generate_si_training_set(
            #     max_input_size, NUM_TEST_SAMPLES, reserved_inputs, frontier_size, branch_size, False, True, 1.0, 1)
            # print('Done. Throughput: {} examples/s'.format(NUM_TEST_SAMPLES / (time.perf_counter() - gen_eval_start_time)))
            # for i in range(sel_inputs.shape[0]):
            #     reserved_inputs.add(tuple([x for x in sel_inputs[i,:] if x != PADDING_TOKEN]))
            # if frontier_size == 4 and branch_size == 4:
            #     eval_inputs_sel, eval_outputs_sel = sel_inputs, sel_outputs

            print('Reserving OOD test data (inference) for frontier_size = {}, branch_size = {}'.format(frontier_size, branch_size))
            stdout.flush()
            inf_inputs, inf_outputs, inf_labels, _ = generator.generate_si_training_set(
                max_input_size, NUM_TEST_SAMPLES, reserved_inputs, frontier_size, branch_size, False, True, 1.0, 2)
            for i in range(inf_inputs.shape[0]):
                if inf_labels[i] != -1:
                    reserved_inputs.add(tuple([x for x in inf_inputs[i, :] if x != PADDING_TOKEN]))
            if frontier_size == 4 and branch_size == 4:
                eval_inputs_inf, eval_outputs_inf = inf_inputs, inf_outputs

    else:
        print('ERROR: Unrecognized task "{}".'.format(task))
        stdout.flush()
        return

    # if BATCH_SIZE < eval_inputs_sel.shape[0]:
    #     eval_inputs_sel = eval_inputs_sel[:BATCH_SIZE]
    #     eval_outputs_sel = eval_outputs_sel[:BATCH_SIZE]

    if BATCH_SIZE < eval_inputs_inf.shape[0]:
        eval_inputs_inf = eval_inputs_inf[:BATCH_SIZE]
        eval_outputs_inf = eval_outputs_inf[:BATCH_SIZE]

    train_filename = 'train{}_v3_inputsize{}_maxlookahead{}_{}seed{}.pkl'.format(dataset_size, max_input_size, max_lookahead, 'padded_' if add_padding else '', seed_value)

    prefix = 'si_split_results/'

    if not torch.cuda.is_available():
        print("ERROR: CUDA device is not available.")
        #from sys import exit
        #exit(-1)
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # compute the checkpoint filenames and try to resume from the last one
    filename = prefix + 'checkpoints_v3_{}layer_inputsize{}_maxlookahead{}_seed{}_train{}'.format(nlayers, max_input_size, max_lookahead, seed_value, dataset_size if dataset_size != -1 else 'streaming')
    if hidden_dim != 16:
        filename += '_hiddendim{}'.format(hidden_dim)
    if bidirectional:
        filename += '_nomask'
    if pos_emb == 'none':
        filename += '_NoPE'
    elif pos_emb == 'rotary':
        filename += '_RoPE'
    if learnable_token_emb:
        filename += '_learntokemb'
    if ablate == "none":
        filename += '_unablated'
    elif ablate == "attn_linear":
        filename += '_ablateattnlinear'
    if toeplitz_attn:
        filename += '_toeplitz'
        if toeplitz_pos_only:
            filename += 'pos'
    if toeplitz_reg != 0.0:
        filename += '_toeplitz'
        if toeplitz_pos_only:
            filename += 'pos'
        filename += str(toeplitz_reg)
    if not pre_ln:
        filename += '_postLN'
    if add_padding:
        filename += '_padded'
    if curriculum_mode == 'y':
        filename += '_curriculum'
    elif curriculum_mode == 'layerbylayer':
        filename += '_layercurriculum'
    elif curriculum_mode == 'layerbylayer2':
        filename += '_layercurriculum2'
    if looped:
        filename += '_looped'
    if task != 'search':
        filename += '_' + task
    if distribution != 'crafted':
        filename += '_' + distribution.replace('_', '-')
    if nhead != 1:
        filename += '_nhead' + str(nhead)
    if warm_up != 0:
        filename += '_warmup' + str(warm_up)
    if batch_size != 2**8:
        filename += '_batchsize' + str(batch_size)
    if learning_rate != 1.0e-5:
        filename += '_lr' + str(learning_rate)
    if update_rate != 2 ** 18:
        filename += '_update' + str(update_rate)
    if reinforce == 'y':
        filename += '_reinforce'
    if loss == "bce":
        filename += '_bce'

    if isdir(filename):
        existing_epochs = [int(ckpt[(ckpt.rfind('epoch') + len('epoch')):-len('.pt')]) for ckpt in listdir(filename) if ckpt.startswith('epoch')]
    else:
        existing_epochs = []
        makedirs(filename)

    ntoken = (max_input_size-5) // 3 + 5
    d_hid = ntoken + hidden_dim
    dropout = 0
    if ablate == "none":
        ablation_mode = AblationMode.NO_ABLATION
    elif ablate == "attn_linear":
        ablation_mode = AblationMode.ABLATE_ATTN_LINEAR
    elif ablate == "attn_linear_projv":
        ablation_mode = AblationMode.ABLATE_ATTN_LINEAR_PROJV
    if toeplitz_attn and toeplitz_pos_only:
        toeplitz = ToeplitzMode.LOWER_RIGHT
    elif toeplitz_attn and not toeplitz_pos_only:
        toeplitz = ToeplitzMode.BLOCK
    else:
        toeplitz = ToeplitzMode.NONE
    if pos_emb == "absolute":
        pos_emb_mode = PositionEmbedding.ABSOLUTE
    elif pos_emb == "rotary":
        pos_emb_mode = PositionEmbedding.ROTARY
    else:
        pos_emb_mode = PositionEmbedding.NONE

    if len(existing_epochs) == 0:
        if curriculum_mode in ('layerbylayer','layerbylayer2'):
            initial_layers = min(3, nlayers)
        else:
            initial_layers = nlayers

        # For selection inference task with split models
        if task == 'si' and split_si == 'y':
            print("Building an inference model.")
            # model_sel = Transformer(
            #     layers=nlayers,
            #     pad_idx=PADDING_TOKEN,
            #     words=ntoken,
            #     seq_len=max_input_size,
            #     heads=nhead,
            #     dims=max(ntoken, d_hid),
            #     rate=1,
            #     dropout=dropout,
            #     bidirectional=bidirectional,
            #     pos_emb=pos_emb_mode,
            #     learn_token_emb=learnable_token_emb,
            #     ablate=ablation_mode,
            #     toeplitz=toeplitz,
            #     pre_ln=pre_ln,
            #     looped=looped
            # )
            model_inf = Transformer(
                layers=nlayers,
                pad_idx=PADDING_TOKEN,
                words=ntoken,
                seq_len=max_input_size,
                heads=nhead,
                dims=max(ntoken, d_hid),
                rate=1,
                dropout=dropout,
                bidirectional=bidirectional,
                pos_emb=pos_emb_mode,
                learn_token_emb=learnable_token_emb,
                ablate=ablation_mode,
                toeplitz=toeplitz,
                pre_ln=pre_ln,
                looped=looped
            )
            # model_sel.to(device)
            model_inf.to(device)

            if reinforce == 'y':
                # Attach value heads for PPO
                print("Attaching value head")
                # model_sel.value_head = nn.Linear(ntoken, 1).to(device)
                model_inf.value_head = nn.Linear(ntoken, 1).to(device)
        epoch = 0
    else:
        # TODO: this doesnt work currently
        print("Resuming training from file currently not supported for split SI models")
        exit(1)
        last_epoch = max(existing_epochs)
        epoch = last_epoch + 1
        print("Loading model from '{}/epoch{}.pt'...".format(filename, last_epoch))
        stdout.flush()
        loaded_obj = torch.load(filename + '/epoch{}.pt'.format(last_epoch), map_location=device)
        model, random_state, np_random_state, torch_random_state = loaded_obj
        setstate(random_state)
        np.random.set_state(np_random_state)
        torch.set_rng_state(torch_random_state.cpu())

    if loss == "bce":
        loss_func = BCEWithLogitsLoss(reduction='mean')
    else:
        loss_func = CrossEntropyLoss(ignore_index=PADDING_TOKEN, reduction='mean')
    INITIAL_LR = 1.0e-4
    TARGET_LR = learning_rate

    if task == 'si' and split_si == 'y':
        # optimizer_sel = SophiaG((p for p in model_sel.parameters() if p.requires_grad), lr=learning_rate, weight_decay=0.1)
        optimizer_inf = SophiaG((p for p in model_inf.parameters() if p.requires_grad), lr=learning_rate, weight_decay=0.1)

    log_interval = 1
    eval_interval = 1
    save_interval = 1

    if curriculum_mode == 'n':
        initial_lookahead = max_lookahead
        initial_max_edges = (max_input_size - 5) // 3
        if task == 'si':
            curriculum_alpha = 1.0

    elif curriculum_mode == 'y':
        initial_lookahead = 1
        initial_max_edges = (max_input_size - 5) // 3

        # For SI curriculum learning
        if task == 'si':
            curriculum_alpha = 0.1

    if hasattr(model_inf, 'lookahead'):
        initial_lookahead = model_inf.lookahead
    else:
        # model_sel.lookahead = initial_lookahead
        model_inf.lookahead = initial_lookahead

    if hasattr(model_inf, 'max_edges'):
        initial_max_edges = model_inf.max_edges
    else:
        # model_sel.max_edges = initial_max_edges
        model_inf.max_edges = initial_max_edges

    # For SI curriculum
    if hasattr(model_inf, 'alpha'):
        curriculum_alpha = model_inf.alpha
    else:
        # model_sel.alpha = curriculum_alpha
        model_inf.alpha = curriculum_alpha

    if dataset_size == -1:
        # we are doing streaming training, so use an IterableDataset
        from itertools import cycle
        from threading import Lock
        STREAMING_BLOCK_SIZE = update_rate
        NUM_DATA_WORKERS = 2
        seed_generator = Random(seed_value)
        seed_generator_lock = Lock()
        seed_values = []

        def get_seed(index):
            if index < len(seed_values):
                return seed_values[index]
            seed_generator_lock.acquire()
            while index >= len(seed_values):
                seed_values.append(seed_generator.randrange(2 ** 32))
            seed_generator_lock.release()
            return seed_values[index]


        class StreamingDatasetSI(torch.utils.data.IterableDataset):
            def __init__(self, offset, lookahead, max_edges, alpha, sample_type):
                super(StreamingDatasetSI).__init__()
                self.offset = offset
                self.lookahead = lookahead
                self.max_edges = max_edges

                # For SI curriculum
                self.alpha = alpha
                # For SI split
                self.sample_type = sample_type

                self.multiprocessing_manager = multiprocessing.Manager()
                self.total_collisions = self.multiprocessing_manager.Value(int, 0)
                self.collisions_lock = self.multiprocessing_manager.Lock()

            def process_data(self, start):
                current = start
                worker_info = torch.utils.data.get_worker_info()
                worker_id = worker_info.id
                max_prefix_vertices = (0 if distribution == 'crafted_no_prefix' else max_input_size)
                while True:
                    worker_start_time = time.perf_counter()
                    new_seed = get_seed(current)
                    generator.set_seed(new_seed)
                    seed(new_seed)
                    torch.manual_seed(new_seed)
                    np.random.seed(new_seed)

                    generate_start_time = time.perf_counter()
                    # Update code to include curiculum training
                    if curriculum_mode == 'y':
                        curr_max_edges = int(self.alpha * (max_input_size - 2) / 6)
                        curr_max_frontier = (curr_max_edges + 1) // 2
                        curr_max_branch = curr_max_edges

                        if reinforce == 'y':
                            batch = generator.generate_si_training_set_reward(
                                max_input_size, BATCH_SIZE, reserved_inputs, curr_max_frontier, curr_max_branch,
                                True, True, self.alpha, self.sample_type)
                        else:
                            batch = generator.generate_si_training_set(
                                max_input_size, BATCH_SIZE, reserved_inputs, curr_max_frontier, curr_max_branch,
                                True, True, self.alpha, self.sample_type)

                    else:
                        # Call the updated generator with sample_type parameter.
                        if reinforce == 'y':
                            batch = generator.generate_si_training_set_reward(
                                max_input_size, BATCH_SIZE, reserved_inputs,
                                max_frontier_size, max_branch_size, True, True, self.alpha,
                                self.sample_type)
                            # batch returns (inputs, outputs, labels, rewards, num_collisions)
                        else:
                            batch = generator.generate_si_training_set(
                                max_input_size, BATCH_SIZE, reserved_inputs,
                                max_frontier_size, max_branch_size, True, True, self.alpha,
                                self.sample_type)
                            # batch returns (inputs, outputs, labels, num_collisions)

                    if reinforce == 'y':
                        if batch[4] != 0:
                            with self.collisions_lock:
                                self.total_collisions.value += batch[4]
                            stdout.flush()
                    else:
                        if batch[3] != 0:
                            with self.collisions_lock:
                                self.total_collisions.value += batch[3]
                            stdout.flush()

                    worker_end_time = time.perf_counter()
                    #print('[WORKER {}] yield = {}, throughput = {} examples/s, rank = {}'.format(worker_id, current, BATCH_SIZE / (worker_end_time - worker_start_time), multiprocessing.current_process()._identity[0]))
                    #print('[WORKER {}] time to get seed = {}s, time to generate data = {}s'.format(worker_id, generate_start_time - worker_start_time, worker_end_time - generate_start_time))
                    #stdout.flush()
                    yield batch[:-1]
                    current += NUM_DATA_WORKERS

            def __iter__(self):
                worker_info = torch.utils.data.get_worker_info()
                worker_id = worker_info.id
                return self.process_data(self.offset + worker_id)

        # sel_dataset = StreamingDatasetSI(epoch * STREAMING_BLOCK_SIZE // BATCH_SIZE, model_sel.lookahead, model_sel.max_edges, model_sel.alpha, 1)
        inf_dataset = StreamingDatasetSI(epoch * STREAMING_BLOCK_SIZE // BATCH_SIZE, model_inf.lookahead, model_inf.max_edges, model_inf.alpha, 2)
        # sel_loader = DataLoader(sel_dataset, batch_size=None, num_workers=NUM_DATA_WORKERS, pin_memory=True, prefetch_factor=8)
        inf_loader = DataLoader(inf_dataset, batch_size=None, num_workers=NUM_DATA_WORKERS, pin_memory=True, prefetch_factor=8)

    examples_seen = epoch * STREAMING_BLOCK_SIZE
    LR_DECAY_TIME = 2**24 # examples seen
    if reinforce == 'y':
        while True:
            start_time = time.perf_counter()
            transfer_time = 0.0
            train_time = 0.0
            log_time = 0.0
            epoch_loss = 0.0
            # epoch_loss_sel = 0.0
            epoch_loss_inf = 0.0
            num_batches = 0
            effective_dataset_size = (STREAMING_BLOCK_SIZE if dataset_size == -1 else dataset_size)
            # reinit_data_loader_sel = False
            reinit_data_loader_inf = False
            for inf_batch in inf_loader:
                batch_start_time = time.perf_counter()

                if warm_up != 0:
                    if examples_seen < warm_up:
                        lr = examples_seen * INITIAL_LR / warm_up
                    elif examples_seen < warm_up + LR_DECAY_TIME:
                        lr = (0.5 * np.cos(np.pi * (examples_seen - warm_up) / LR_DECAY_TIME) + 0.5) * (
                                    INITIAL_LR - TARGET_LR) + TARGET_LR
                    else:
                        lr = TARGET_LR
                else:
                    lr = TARGET_LR
                # for param_group in optimizer_sel.param_groups:
                #     param_group['lr'] = lr
                for param_group in optimizer_inf.param_groups:
                    param_group['lr'] = lr

                # model_sel.train()
                model_inf.train()

                # sel_inputs, sel_outputs, sel_labels, sel_rewards = sel_batch
                inf_inputs, inf_outputs, inf_labels, inf_rewards = inf_batch

                # sel_inputs = sel_inputs.to(device, non_blocking=True)
                inf_inputs = inf_inputs.to(device, non_blocking=True)
                # sel_outputs = sel_outputs.to(device, non_blocking=True)
                inf_outputs = inf_outputs.to(device, non_blocking=True)
                # sel_labels = sel_labels.to(device, non_blocking=True)
                inf_labels = inf_labels.to(device, non_blocking=True)
                # sel_rewards = torch.tensor(sel_rewards, device=device, dtype=torch.float32)
                inf_rewards = torch.tensor(inf_rewards, device=device, dtype=torch.float32)
                examples_seen += BATCH_SIZE

                # Get current policy outputs and compute old log probs and values.
                with torch.no_grad():
                    # logits_sel, values_sel = get_policy_and_value(model_sel, sel_inputs)
                    # prob_dist_sel = torch.softmax(logits_sel, dim=1)
                    # old_log_probs_sel = torch.log(torch.gather(prob_dist_sel, 1, sel_outputs.unsqueeze(1)).squeeze(1) + 1e-8)

                    logits_inf, values_inf = get_policy_and_value(model_inf, inf_inputs)
                    prob_dist_inf = torch.softmax(logits_inf, dim=1)
                    old_log_probs_inf = torch.log(
                        torch.gather(prob_dist_inf, 1, inf_outputs.unsqueeze(1)).squeeze(1) + 1e-8)

                    # Run several PPO epochs over the batch.
                    # ppo_loss_sel = 0.0
                    ppo_loss_inf = 0.0
                    batch_size = inf_inputs.size(0)
                    num_minibatches = max(1, batch_size // PPO_MINIBATCH_SIZE)
                    for epoch_ppo in range(PPO_EPOCHS):
                        # Shuffle indices
                        indices = torch.randperm(batch_size)
                        for i in range(num_minibatches):
                            minibatch_idx = indices[i * PPO_MINIBATCH_SIZE: (i + 1) * PPO_MINIBATCH_SIZE]

                            # mb_inputs_sel = sel_inputs[minibatch_idx]
                            # mb_actions_sel = sel_labels[minibatch_idx]
                            # mb_old_log_probs_sel = old_log_probs_sel[minibatch_idx]
                            # mb_rewards_sel = sel_rewards[minibatch_idx]
                            # mb_old_values_sel = values_sel[minibatch_idx]
                            #
                            # loss_vals_sel = ppo_update(model_sel, optimizer_sel, mb_inputs_sel, mb_actions_sel,
                            #                            mb_old_log_probs_sel, mb_rewards_sel, mb_old_values_sel)
                            # ppo_loss_sel += loss_vals_sel[0]

                            mb_inputs_inf = inf_inputs[minibatch_idx]
                            mb_actions_inf = inf_labels[minibatch_idx]
                            mb_old_log_probs_inf = old_log_probs_inf[minibatch_idx]
                            mb_rewards_inf = inf_rewards[minibatch_idx]
                            mb_old_values_inf = values_inf[minibatch_idx]

                            loss_vals_inf = ppo_update(model_inf, optimizer_inf, mb_inputs_inf, mb_actions_inf,
                                                       mb_old_log_probs_inf, mb_rewards_inf, mb_old_values_inf)
                            ppo_loss_inf += loss_vals_inf[0]

                    # ppo_loss_sel /= (PPO_EPOCHS * num_minibatches)
                    ppo_loss_inf /= (PPO_EPOCHS * num_minibatches)
                    # epoch_loss_sel += ppo_loss_sel
                    epoch_loss_inf += ppo_loss_inf
                    num_batches += 1

                    if num_batches % 100 == 0:
                        print(f"Batch {num_batches}: PPO inf loss = {ppo_loss_inf:.4f}")
                        elapsed_time = time.perf_counter() - start_time
                        print("Epoch {}: avg PPO inf loss = {:.4f}, throughput = {:.2f} examples/s"
                              .format(epoch, epoch_loss_inf / num_batches,
                                      (BATCH_SIZE * num_batches) / elapsed_time))

                # Evaluate on OOD data (using standard evaluation function)
                # model_sel.eval()
                model_inf.eval()
                # sel_test_acc, sel_test_loss, _ = evaluate_model(model_sel, eval_inputs_sel, eval_outputs_sel)
                inf_test_acc, inf_test_loss, _ = evaluate_model(model_inf, eval_inputs_inf, eval_outputs_inf)
                # print("Epoch {}: Test Sel Acc = {:.2f}, Loss = {:.6f}".format(epoch, sel_test_acc, sel_test_loss))
                print("Epoch {}: Test Inf Acc = {:.2f}, Loss = {:.6f}".format(epoch, inf_test_acc, inf_test_loss))
                # Save checkpoints
                # sel_ckpt_filename = filename + '/sel_epoch{}.pt'.format(epoch)
                inf_ckpt_filename = filename + '/inf_epoch{}.pt'.format(epoch)
                print('Saving models to "{}".'.format(inf_ckpt_filename))
                # torch.save((model_sel, getstate(), np.random.get_state(), torch.get_rng_state()), sel_ckpt_filename)
                torch.save((model_inf, getstate(), np.random.get_state(), torch.get_rng_state()), inf_ckpt_filename)
                epoch += 1

    else:
        while True:
            start_time = time.perf_counter()
            transfer_time = 0.0
            train_time = 0.0
            log_time = 0.0
            epoch_loss = 0.0
            # epoch_loss_sel = 0.0
            epoch_loss_inf = 0.0
            num_batches = 0
            effective_dataset_size = (STREAMING_BLOCK_SIZE if dataset_size == -1 else dataset_size)
            # reinit_data_loader_sel = False
            reinit_data_loader_inf = False
            for inf_batch in inf_loader:
                batch_start_time = time.perf_counter()

                if warm_up != 0:
                    if examples_seen < warm_up:
                        lr = examples_seen * INITIAL_LR / warm_up
                    elif examples_seen < warm_up + LR_DECAY_TIME:
                        lr = (0.5*np.cos(np.pi * (examples_seen - warm_up) / LR_DECAY_TIME) + 0.5) * (INITIAL_LR - TARGET_LR) + TARGET_LR
                    else:
                        lr = TARGET_LR
                else:
                    lr = TARGET_LR
                # for param_group in optimizer_sel.param_groups:
                #     param_group['lr'] = lr
                for param_group in optimizer_inf.param_groups:
                    param_group['lr'] = lr

                # model_sel.train()
                model_inf.train()

                # sel_inputs, sel_outputs, sel_labels = sel_batch
                inf_inputs, inf_outputs, inf_labels = inf_batch

                # sel_inputs = sel_inputs.to(device, non_blocking=True)
                inf_inputs = inf_inputs.to(device, non_blocking=True)
                # sel_outputs = sel_outputs.to(device, non_blocking=True)
                inf_outputs = inf_outputs.to(device, non_blocking=True)
                # sel_labels = sel_labels.to(device, non_blocking=True)
                inf_labels = inf_labels.to(device, non_blocking=True)
                examples_seen += BATCH_SIZE

                #if device.type == 'cuda':
                #	torch.cuda.synchronize(device)
                train_start_time = time.perf_counter()
                transfer_time += train_start_time - batch_start_time

                # Forward pass for selection model.
                # logits_sel = model_sel(sel_inputs)
                # loss_sel = loss_func(logits_sel[:, -1, :], sel_labels)

                # Forward pass for inference model.
                logits_inf = model_inf(inf_inputs)

                if loss == "bce":
                    loss_inf = loss_func(logits_inf[:, -1, :], inf_outputs)
                else:
                    loss_inf = loss_func(logits_inf[:, -1, :], inf_labels)

                loss_val = loss_inf

                # if toeplitz_reg != 0.0:
                #     def compute_toeplitz_regularization(m):
                #         regularization = 0.0
                #         for i in range(-A.size(0) + 1, A.size(1)):
                #             regularization += torch.var(torch.diagonal(A, offset=i), unbiased=False)
                #         return regularization
                #
                #     for transformer in model.transformers:
                #         P_q = next(v for k,v in transformer.attn.proj_q.named_parameters() if k == 'weight')
                #         P_k = next(v for k,v in transformer.attn.proj_k.named_parameters() if k == 'weight')
                #         A = torch.matmul(P_q.transpose(-2,-1),P_k)
                #         if not toeplitz_pos_only:
                #             loss_val += toeplitz_reg * compute_toeplitz_regularization(A[:ntoken,:ntoken])
                #             loss_val += toeplitz_reg * compute_toeplitz_regularization(A[:ntoken,ntoken:d_hid])
                #             loss_val += toeplitz_reg * compute_toeplitz_regularization(A[:ntoken,d_hid:])
                #             loss_val += toeplitz_reg * compute_toeplitz_regularization(A[ntoken:d_hid,:ntoken])
                #             loss_val += toeplitz_reg * compute_toeplitz_regularization(A[ntoken:d_hid,ntoken:d_hid])
                #             loss_val += toeplitz_reg * compute_toeplitz_regularization(A[ntoken:d_hid,d_hid:])
                #             loss_val += toeplitz_reg * compute_toeplitz_regularization(A[d_hid:,:ntoken])
                #             loss_val += toeplitz_reg * compute_toeplitz_regularization(A[d_hid:,ntoken:d_hid])
                #         loss_val += toeplitz_reg * compute_toeplitz_regularization(A[d_hid:,d_hid:])

                # epoch_loss_sel += loss_sel.item()
                epoch_loss_inf += loss_inf.item()
                epoch_loss += loss_val.item()

                # loss_val.backward()
                # loss_sel.backward()
                loss_inf.backward()

                if examples_seen % (BATCH_SIZE * grad_accumulation_steps) == 0:
                    # optimizer_sel.step()
                    # optimizer_sel.zero_grad()
                    optimizer_inf.step()
                    optimizer_inf.zero_grad()

                #if device.type == 'cuda':
                #	torch.cuda.synchronize(device)
                log_start_time = time.perf_counter()
                train_time += log_start_time - train_start_time
                num_batches += 1

                if num_batches == effective_dataset_size // BATCH_SIZE:
                    #time4 = time.perf_counter()
                    #print('[MAIN] Time to train: {}s'.format(time4 - time3))
                    #stdout.flush()

                    if epoch % log_interval == 0:
                        elapsed_time = time.perf_counter() - start_time
                        # avg_loss_sel = epoch_loss_sel / num_batches
                        avg_loss_inf = epoch_loss_inf / num_batches
                        avg_loss_combined = epoch_loss / num_batches
                        print("epoch = {}, training inf loss = {}".format(epoch, avg_loss_inf))
                        if device.type == 'cuda':
                            utilization = popen('nvidia-smi --query-gpu=utilization.gpu --format=csv').read().split('\n')[1]
                            print("throughput = {} examples/s, GPU utilization = {}".format(effective_dataset_size / elapsed_time, utilization))
                        else:
                            print("throughput = {} examples/s".format(effective_dataset_size / elapsed_time))
                        print('Total number of training examples generated that are in the test set: {}'.format(inf_dataset.total_collisions.value))
                        print('Learning rate: {}'.format(lr))
                        print("[PROFILE] Total batch time: {}s".format(elapsed_time))
                        print("[PROFILE] Time to transfer data to GPU: {}s".format(transfer_time))
                        print("[PROFILE] Time to train: {}s".format(train_time))
                        print("[PROFILE] Time to log/save/validate: {}s".format(log_time))
                        stdout.flush()
                        start_time = time.perf_counter()
                        transfer_time = 0.0
                        train_time = 0.0
                        log_time = 0.0

                    if epoch % eval_interval == 0:
                        # model_sel.eval()
                        model_inf.eval()

                        # sel_logits, _ = model_sel(sel_inputs)
                        inf_logits, _ = model_inf(inf_inputs)

                        # print(f"Sel in: {sel_inputs},\n out: {sel_outputs},\n sum: {torch.sum(sel_outputs, dim=1)}")
                        # print(f"Inf: {inf_inputs},\n out: {inf_outputs},\n sum: {torch.sum(inf_outputs, dim=1)}")

                        # sel_training_acc = torch.sum(torch.gather(sel_outputs, 1, torch.argmax(sel_logits[:,-1,:],dim=1).unsqueeze(1))).item() / sel_outputs.size(0)
                        inf_training_acc = torch.sum(torch.gather(inf_outputs, 1, torch.argmax(inf_logits[:,-1,:],dim=1).unsqueeze(1))).item() / inf_outputs.size(0)


                        # print("training sel accuracy: %.2f±%.2f" % (sel_training_acc, binomial_confidence_int(sel_training_acc, sel_outputs.size(0))))
                        print("training inf accuracy: %.2f±%.2f" % (inf_training_acc, binomial_confidence_int(inf_training_acc, inf_outputs.size(0))))
                        del inf_inputs, inf_outputs, inf_labels
                        stdout.flush()

                        # sel_test_acc, sel_test_loss, _ = evaluate_model(model_sel, eval_inputs_sel, eval_outputs_sel)
                        inf_test_acc, inf_test_loss, _ = evaluate_model(model_inf, eval_inputs_inf, eval_outputs_inf)

                        # print("Epoch {}: Test Selection Acc = {:.2f}±{:.2f}, Loss = {:.6f}".format(
                        #     epoch, sel_test_acc, binomial_confidence_int(sel_test_acc, 1000), sel_test_loss))
                        print("Epoch {}: Test Inference Acc = {:.2f}±{:.2f}, Loss = {:.6f}".format(
                            epoch, inf_test_acc, binomial_confidence_int(inf_test_acc, 1000), inf_test_loss))
                        stdout.flush()
                        #time6 = time.perf_counter()
                        #print('[MAIN] Time to evaluate model: {}s'.format(time6 - time5))
                        #stdout.flush()

                        # Add code for curriculum training for SI
                        # if task == 'si' and curriculum_mode == 'y' and sel_training_acc > 0.98:
                        #     if model_sel.alpha < 1.0:
                        #         model_sel.alpha = min(1.0, model_sel.alpha + 0.1)
                        #         print("Curriculum update: selection alpha increased to {:.2f}.".format(model_sel.alpha))
                        #         reinit_data_loader_sel = True
                        #         break

                        if task == 'si' and curriculum_mode == 'y' and inf_training_acc > 0.98:
                            if model_inf.alpha < 1.0:
                                model_inf.alpha = min(1.0, model_inf.alpha + 0.1)
                                print("Curriculum update: inference alpha increased to {:.2f}.".format(model_inf.alpha))
                                reinit_data_loader_inf = True
                                break

                    if epoch % save_interval == 0:
                        # sel_ckpt_filename = filename + '/sel_epoch{}.pt'.format(epoch)
                        inf_ckpt_filename = filename + '/inf_epoch{}.pt'.format(epoch)
                        print('saving to "{}".'.format(inf_ckpt_filename))
                        # torch.save((model_sel,getstate(),np.random.get_state(),torch.get_rng_state()), sel_ckpt_filename)
                        torch.save((model_inf,getstate(),np.random.get_state(),torch.get_rng_state()), inf_ckpt_filename)
                        print('done saving models.')
                        stdout.flush()

                    #time5 = time.perf_counter()
                    #print('[MAIN] Time to save model: {}s'.format(time5 - time4))
                    #stdout.flush()

                    #time7 = time.perf_counter()
                    #print('[MAIN] Total time for epoch: {}s'.format(time7 - time1))
                    #stdout.flush()
                    epoch += 1
                    num_batches = 0
                    epoch_loss = 0.0
                    epoch_loss_inf = 0.0
                    # epoch_loss_sel = 0.0
                    if reinit_data_loader_inf:
                        break

                #if device.type == 'cuda':
                #	torch.cuda.synchronize(device)
                log_end_time = time.perf_counter()
                log_time += log_end_time - log_start_time

            # if reinit_data_loader_sel:
            #     sel_dataset = StreamingDatasetSI(epoch * STREAMING_BLOCK_SIZE // BATCH_SIZE, model_sel.lookahead,
            #                                      model_sel.max_edges, model_sel.alpha, 1)
            #     sel_loader = DataLoader(sel_dataset, batch_size=None, num_workers=NUM_DATA_WORKERS, pin_memory=True,
            #                             prefetch_factor=8)
            #     reinit_data_loader_sel = False

            if reinit_data_loader_inf:
                inf_dataset = StreamingDatasetSI(epoch * STREAMING_BLOCK_SIZE // BATCH_SIZE, model_inf.lookahead,
                                                 model_inf.max_edges, model_inf.alpha, 2)
                inf_loader = DataLoader(inf_dataset, batch_size=None, num_workers=NUM_DATA_WORKERS, pin_memory=True,
                                        prefetch_factor=8)
                reinit_data_loader_inf = False


if __name__ == "__main__":
    import argparse
    def parse_bool_arg(v):
        if isinstance(v, bool):
            return v
        elif v.lower() in ('yes', 'true', 'y', 't', '1'):
            return True
        elif v.lower() in ('no', 'false', 'n', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-input-size", type=int)
    parser.add_argument("--dataset-size", type=int)
    parser.add_argument("--max-lookahead", type=int, required=False)
    parser.add_argument("--nlayers", type=int)
    parser.add_argument("--nhead", type=int, default=1, required=False)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bidirectional", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--pos-emb", type=str, required=True, choices=["absolute", "rotary", "none"])
    parser.add_argument("--learn-tok-emb", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--toeplitz-attn", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--toeplitz-reg", type=float, required=True, default=0.0)
    parser.add_argument("--toeplitz-pos-only", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--add-padding", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--ablate", type=str, default="none", choices=["none", "attn_linear", "attn_linear_projv"])
    parser.add_argument("--preLN", type=parse_bool_arg, required=True, metavar="'y/n'")
    parser.add_argument("--curriculum", type=str, required=True, choices=["y", "n", "layerbylayer", "layerbylayer2"])
    parser.add_argument("--looped", type=parse_bool_arg, default=False)
    parser.add_argument("--task", type=str, default="search", choices=["search", "dfs", "si"])
    parser.add_argument("--distribution", type=str, default="crafted", choices=["simple", "crafted", "crafted_no_prefix", "star"])
    parser.add_argument("--warm-up", type=int, default=0, required=False)
    parser.add_argument("--batch-size", type=int, default=2**8, required=False)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5, required=False)
    parser.add_argument("--update-rate", type=int, default=2**18, required=False)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1, required=False)
    parser.add_argument('--split-si', type=str, default='n', choices=['y', 'n'], required=False)
    parser.add_argument('--reinforce', type=str, default='n', choices=['y', 'n'], required=False)
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'bce'], required=False)
    args = parser.parse_args()

    train(
        max_input_size=args.max_input_size,
        dataset_size=args.dataset_size,
        distribution=args.distribution,
        max_lookahead=args.max_lookahead,
        seed_value=args.seed,
        nhead=args.nhead,
        nlayers=args.nlayers,
        hidden_dim=args.hidden_dim,
        bidirectional=args.bidirectional,
        pos_emb=args.pos_emb,
        learnable_token_emb=args.learn_tok_emb,
        toeplitz_attn=args.toeplitz_attn,
        toeplitz_reg=args.toeplitz_reg,
        toeplitz_pos_only=args.toeplitz_pos_only,
        add_padding=args.add_padding,
        ablate=args.ablate,
        pre_ln=args.preLN,
        curriculum_mode=args.curriculum,
        looped=args.looped,
        task=args.task,
        warm_up=args.warm_up,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        update_rate=args.update_rate,
        grad_accumulation_steps=args.grad_accumulation_steps,
        split_si=args.split_si,
        reinforce=args.reinforce,
        loss=args.loss)

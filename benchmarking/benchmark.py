import os
import random
import sys
import sysconfig
from io import StringIO
import asyncio

from openai import AsyncOpenAI
aclient = None
from together import AsyncTogether

from faker import Faker
from openai import OpenAI
from pybind11.__main__ import print_includes

# Add the parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from train import generate_example

# Set working path to the parent
os.chdir("..")

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
		sys.exit(1)

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


def generate_graph_text(
	max_input_size,
	num_vertices,
	max_num_parents,
	lookahead,
	num_paths,
):

	# # Get graph using generate_example
	# graph, start, end, paths = generate_example(
	# 	num_vertices,
	# 	max_num_parents,
	# 	(max_input_size - 5) // 3,
	# 	get_shortest_paths=True,
	# 	lookahead=lookahead,
	# 	num_paths=num_paths,
	# 	max_prefix_vertices=0
	# )

	# Get graph using the C++ generator
	inputs, outputs, labels, num_collisions = generator.generate_training_set(
		max_input_size,
		1,	# batch size
		lookahead,
		max_num_parents * num_vertices, # max edges
		set(), # reserved vertices
		-1, # distance from start
		0, # max_prefix_vertices
		True, # quiet
		num_paths # number of paths
	)

	# If the random DAG fails to generate a valid example, return None
	if inputs is None or outputs is None:
		return None

	# Token IDs
	PADDING_TOKEN      = (max_input_size - 5) // 3 + 3
	EDGE_PREFIX_TOKEN  = (max_input_size - 5) // 3 + 2
	PATH_PREFIX_TOKEN  = (max_input_size - 5) // 3 + 1
	QUERY_PREFIX_TOKEN = (max_input_size - 5) // 3 + 4

	# prefix = []

	# # Add list of edges
	# for vertex in inputs:
	# 	for child in vertex.children:
	# 		prefix.append(EDGE_PREFIX_TOKEN)
	# 		prefix.append(vertex.id)
	# 		prefix.append(child.id)
	#
	# # Add query
	# prefix.append(QUERY_PREFIX_TOKEN)
	# prefix.append(start.id)
	# prefix.append(end.id)
	#
	# # Add path prefix token
	# prefix.append(PATH_PREFIX_TOKEN)
	#
	# # Add the starting vertex as my current position
	# prefix.append(start.id)

	# Function for mapping tokens to letters
	def token_to_str(token):
		if token == EDGE_PREFIX_TOKEN:
			return "E"
		elif token == QUERY_PREFIX_TOKEN:
			return "Q"
		elif token == PATH_PREFIX_TOKEN:
			return "P"
			# return ""
		elif token == PADDING_TOKEN:
			return ""
		else:
			return str(token)

	# Convert all tokens to letters
	text_tokens = [token_to_str(t) for t in inputs[0] if token_to_str(t) != ""]

	final_str = " ".join(text_tokens)
	return final_str, labels[0]


def generate_name(n):
	"""
	Generates n random first names using the Faker library.
	"""
	fake = Faker()
	# unique_count = n // 2 + 1
	# names = [fake.unique.first_name() for _ in range(unique_count)]
	#
	# result = [names[0]]
	# for name in names[1:]:
	# 	result.extend([name, name])
	#
	# print(result)
	#
	# return result

	return [fake.unique.name() for i in range(n)]


def random_syllable():
	"""
	Generates a random syllable based on a randomly chosen pattern.
	The pattern is built from 'C' (consonant) and 'V' (vowel).
	"""
	vowels = "aeiou"
	consonants = "bcdfghjklmnpqrstvwxyz"
	# Syllable patterns (e.g. "CV" means a consonant followed by a vowel)
	patterns = ["CV", "VC", "CVC", "CVV", "CCV", "VCV", "VCC"]
	pattern = random.choice(patterns)

	syllable = ""
	for char in pattern:
		if char == "C":
			syllable += random.choice(consonants)
		elif char == "V":
			syllable += random.choice(vowels)
	return syllable


def generate_fake_noun():
	"""
	Generates a fake noun by concatenating two randomly generated syllables.
	"""
	return random_syllable() + random_syllable()


def generate_fake_nouns(n):
	"""
	Returns a list of n fake nouns.
	"""
	nouns = set()
	while len(nouns) < n:
		nouns.add(generate_fake_noun())
	return list(nouns)


def generate_words(n):
	fake = Faker()
	return fake.words(nb=n)


def logic_paragraph_from_tokens(tokens_str: str, next_step: int, use_diff_names=True):
	# Split string into tokens
	tokens = tokens_str.strip().split()

	edges = []
	queries = []

	# Keep track of nodes in a set
	node_ids = set()

	i = 0
	while i < len(tokens):
		t = tokens[i]

		if t == "E":
			# Expect "E A B"
			if i + 2 >= len(tokens):
				break  # malformed
			A = int(tokens[i + 1])
			B = int(tokens[i + 2])
			edges.append((A, B))
			node_ids.update([A, B])
			i += 3

		elif t == "Q":
			# Expect "Q X Y"
			if i + 2 >= len(tokens):
				break
			X = int(tokens[i + 1])
			Y = int(tokens[i + 2])
			queries.append((X, Y))
			node_ids.update([X, Y])
			i += 3

		elif t == "P":
			# "P" is just a path prefix token; ignore
			i += 1

		else:
			# Possibly a stray token or something else (like a vertex ID alone).
			i += 1

	# Generate fake names and fake adjectives
	num_nodes = len(node_ids)
	all_names = generate_name(num_nodes)
	all_adjs = generate_fake_nouns(num_nodes)

	sorted_nodes = sorted(node_ids)
	id_to_pair = {}  # node_id -> (name, adjective)
	# Assign all nodes the same name if use_diff_names == False
	for idx, node_id in enumerate(sorted_nodes):
		id_to_pair[node_id] = (all_names[idx if use_diff_names else 0], all_adjs[idx])

	# Lines of logic
	lines = ["Given the following list of predicates:"]

	def get_logic_line(name_a: str, name_b: str, adj_a: str, adj_b: str) -> str:
		choices = [
			f"If {name_a} is {adj_a}, then {name_b} is {adj_b}.",
			f"{name_a} is {adj_a} implies {name_b} is {adj_b}.",
			f"{name_b} is {adj_b} is true if {name_a} is {adj_a}.",
			# f"{adj_b} is true if {adj_a} is true.",
			# f"If {adj_a} then {adj_b} is true.",
			# f"If {adj_a} is true then {adj_b}.",
			f"Given {name_a} is {adj_a} then {name_b} is {adj_b}.",
		]

		sentence = random.choice(choices)
		return sentence[0].upper() + sentence[1:]

	# For each edge:  E A B => "If name(A) is adj(A), then name(A) is adj(B)."
	for (A, B) in edges:
		name_A, adj_A = id_to_pair[A]
		name_B, adj_B = id_to_pair[B]
		lines.append(get_logic_line(name_A, name_B, adj_A, adj_B))

	# For each query: Q X Y => "If name(X) is adj(X), prove that name(X) is adj(Y)."
	for (X, Y) in queries:
		name_x, adj_x = id_to_pair[X]
		name_y, adj_y = id_to_pair[Y]
		lines.append(f"\n\nIf {name_x} is {adj_x}, what is the next step to prove that {name_y} is {adj_y}?")

	# Join all lines into one paragraph
	paragraph = " ".join(lines)

	# Get the correct adjective for the next step
	next_step_adj = id_to_pair[next_step]
	next_step_adj = id_to_pair[next_step][1]

	return paragraph, next_step_adj


async def get_response(prompt: str, model: str):
	response = await aclient.chat.completions.create(model=model,
			messages=[{"role": "user", "content": prompt}])
	tokens = response.usage.completion_tokens_details.reasoning_tokens
	return response.choices[0].message.content, tokens

async def get_response_together(prompt: str, model: str):
	response = await aclient.chat.completions.create(
		model=model,
		messages=[{"role": "user", "content": prompt}],
	)
	tokens = response.usage.total_tokens
	return response.choices[0].message.content, tokens


async def main(samples_per_test: int = 3, lookahead_range: list = range(1, 5), num_paths: int = 2, max_num_parents: int = 3, logic: bool = False, seed: int = None, verbose: bool = True, print_prompts: bool = False, model: str = "gpt-4o", submit_prompts: bool = True):
	global aclient
	if submit_prompts:
		if model != "deepseek-ai/DeepSeek-R1":
			aclient = AsyncOpenAI()
		else:
			aclient = AsyncTogether()

	if seed is not None:
		random.seed(seed)
		generator.set_seed(seed)

	prompts = []
	correct_responses = []
	look_ahead_values = []

	for look_ahead in lookahead_range:
		for _ in range(samples_per_test):
			txt, next_step = generate_graph_text(
				max_input_size=max_num_parents * look_ahead * num_paths * 4,
				num_vertices=max_num_parents * look_ahead,
				max_num_parents=max_num_parents,
				lookahead=look_ahead,
				num_paths=num_paths,
			)

			# Create the prompt for the graph search
			prompt = (f"{txt}\nAbove is a representation of a directed graph search problem, "
					  f"where E A B represents an edge from A to B, and Q X Y represents starting from X and ending at Y, "
					  f"find the shortest path. The vertex after P indicates our current position. Respond with only the "
					  f"next vertex on the shortest path from X to Y and nothing else.")

			# Change the prompt to a logic puzzle if the logic option is enabled
			if logic:
				logic, next_step_adj = logic_paragraph_from_tokens(txt, next_step)
				prompt = f"{logic} Respond with only the trait of the next step."

			prompts.append(prompt)
			correct_responses.append(next_step_adj if logic else next_step)
			look_ahead_values.append(look_ahead)

			if print_prompts:
				print(f"Prompt: {prompt}\n")
				print(f"Correct:   {next_step_adj if logic else next_step}\n")

		if not submit_prompts:
			continue

		# Create async tasks to run multiple AI calls at once

		if model != "deepseek-ai/DeepSeek-R1":
			tasks = [asyncio.create_task(get_response(prompt, model)) for prompt in prompts]
			results = await asyncio.gather(*tasks)
		else:
			tasks = [asyncio.create_task(get_response_together(prompt, model)) for prompt in prompts]
			results = await asyncio.gather(*tasks)


		# Keep track of number of tokens and correct responses
		tokens_used = 0
		correct_count = 0
		for response, correct, lav in zip(results, correct_responses, look_ahead_values):
			response_txt, response_tokens = response
			tokens_used += response_tokens

			if logic:
				if response_txt == correct or correct in response_txt:
					correct_count += 1
					if verbose:
						print(f"Correct, look_ahead={lav}, num_paths={num_paths}, max_num_parents={max_num_parents}. Tokens: {response_tokens}")
				else:
					if verbose:
						print(f"Incorrect, response={response_txt}, correct={correct} look_ahead={lav}, num_paths={num_paths}, max_num_parents={max_num_parents}. Tokens: {response_tokens}")
			else:
				try:
					if int(response_txt) == int(correct):
						correct_count += 1
						if verbose:
							print(f"Correct, look_ahead={lav}, num_paths={num_paths}, max_num_parents={max_num_parents}. Tokens: {response_tokens}")
					else:
						if verbose:
							print(f"Incorrect, response={response_txt}, correct={correct} look_ahead={lav}, num_paths={num_paths}, max_num_parents={max_num_parents}. Tokens: {response_tokens}")
				except ValueError:
					# if verbose:
					print(f"\n***Response={response_txt}, gave a value error, \n *correct={correct}, look_ahead={lav}. Tokens: {response_tokens}")

		tokens_used /= samples_per_test
		print(f"look_ahead={look_ahead}, correct={correct_count}, avg_tokens={tokens_used}\n")

		prompts = []
		correct_responses = []
		look_ahead_values = []


def multiplicative_range(start, stop, step):
	result = []
	current = start
	while (step > 1 and current < stop) or (step < 1 and current > stop):
		result.append(current)
		current *= step
	return result


if __name__ == "__main__":
	look_ahead = multiplicative_range(2, 550, 2)
	print(f"look_ahead range={look_ahead}")
	asyncio.run(main(
		# model="o3-mini",
		model="deepseek-ai/DeepSeek-R1",
		samples_per_test=10,
		lookahead_range=look_ahead,
		num_paths=9,
		logic=False,
		verbose=False,
		print_prompts=False,
		submit_prompts=True
	))



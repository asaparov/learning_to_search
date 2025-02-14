import os
import random
import sys
import sysconfig
from io import StringIO
import asyncio

from openai import AsyncOpenAI

aclient = AsyncOpenAI()
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

# try:
# 	from os.path import getmtime
# 	from importlib.util import find_spec
#
# 	generator_spec = find_spec('generator')
# 	if generator_spec == None:
# 		raise ModuleNotFoundError
# 	if getmtime(generator_spec.origin) < getmtime('generator.cpp'):
# 		print("C++ module `generator` is out-of-date. Compiling from source...")
# 		build_module("generator")
# 	import generator
# except ModuleNotFoundError:
# 	print("C++ module `generator` not found. Compiling from source...")
# 	build_module("generator")
# 	import generator
# except ImportError:
# 	print("Error loading C++ module `generator`. Compiling from source...")
# 	build_module("generator")
# 	import generator
# print("C++ module `generator` loaded.")


def generate_graph_text(
	max_input_size,
	num_vertices,
	max_num_parents,
	lookahead,
	num_paths,
):

	# Get graph using generate_example
	graph, start, end, paths = generate_example(
		num_vertices,
		max_num_parents,
		(max_input_size - 5) // 3,
		get_shortest_paths=True,
		lookahead=lookahead,
		num_paths=num_paths,
		max_prefix_vertices=0
	)

	# If the random DAG fails to generate a valid example, return None
	if graph is None or paths is None:
		return None

	# Token IDs
	PADDING_TOKEN      = (max_input_size - 5) // 3 + 3
	EDGE_PREFIX_TOKEN  = (max_input_size - 5) // 3 + 2
	PATH_PREFIX_TOKEN  = (max_input_size - 5) // 3 + 1
	QUERY_PREFIX_TOKEN = (max_input_size - 5) // 3 + 4

	prefix = []

	# Add list of edges
	for vertex in graph:
		for child in vertex.children:
			prefix.append(EDGE_PREFIX_TOKEN)
			prefix.append(vertex.id)
			prefix.append(child.id)

	# Add query
	prefix.append(QUERY_PREFIX_TOKEN)
	prefix.append(start.id)
	prefix.append(end.id)

	# Add path prefix token
	prefix.append(PATH_PREFIX_TOKEN)

	# Add the starting vertex as my current position
	prefix.append(start.id)

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
	text_tokens = [token_to_str(t) for t in prefix if token_to_str(t) != ""]


	final_str = " ".join(text_tokens)
	return final_str, paths


def generate_name(n):
	"""
	Generates n random first names using the Faker library.
	"""
	fake = Faker()
	return [fake.unique.first_name() for i in range(n)]


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


def logic_paragraph_from_tokens(tokens_str: str, paths: list, use_diff_names=False):
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
	lines = []

	# For each edge:  E A B => "If name(A) is adj(A), then name(A) is adj(B)."
	for (A, B) in edges:
		name_A, adj_A = id_to_pair[A]
		_, adj_B = id_to_pair[B]
		lines.append(f"If {name_A} is {adj_A}, then {name_A} is {adj_B}.")

	# For each query: Q X Y => "If name(X) is adj(X), prove that name(X) is adj(Y)."
	for (X, Y) in queries:
		name_x, adj_x = id_to_pair[X]
		_, adj_y = id_to_pair[Y]
		lines.append(f"If {name_x} is {adj_x}, prove that {name_x} is {adj_y}.")

	# Join all lines into one paragraph
	paragraph = " ".join(lines)

	# Convert the shortest path to adjectives
	path_adjectives = [id_to_pair[node.id][1] for node in paths[0]]

	return paragraph, path_adjectives


async def get_response(prompt: str):
	response = await aclient.chat.completions.create(model="gpt-4o",
			messages=[{"role": "user", "content": prompt}])
	return response.choices[0].message.content


async def main():
	num_paths = 2
	max_num_parents = 3

	prompts = []
	correct_responses = []
	look_ahead_values = []

	for look_ahead in range(1, 55):

		print(f"Look ahead: {look_ahead}")
		for _ in range(3):
			txt, path = generate_graph_text(
				max_input_size=max_num_parents * look_ahead * num_paths * 4,
				num_vertices=max_num_parents * look_ahead,
				max_num_parents=max_num_parents,
				lookahead=look_ahead,
				num_paths=num_paths,
			)
			logic, logic_path = logic_paragraph_from_tokens(txt, path)
			path = [node.id for node in path[0]]

			prompt = (f"{txt}\nAbove is a representation of a directed graph search problem, "
					  f"where E A B represents an edge from A to B, and Q X Y represents starting from X and ending at Y,"
					  f"find the shortest path. The vertex after P indicates our current position. Respond with only the "
					  f"next vertex on the shortest path from X to Y.")

			# prompt = f"{logic} Respond with only the list of words that you use in the proof."

			prompts.append(prompt)
			correct_responses.append(path[1])
			look_ahead_values.append(look_ahead)

			print(prompt, "\n")
			# print(logic_path)
			print(path[1])
			print("\n")


		# if len(prompts) >= 9:
		# 	tasks = [asyncio.create_task(get_response(prompt)) for prompt in prompts]
		# 	results = await asyncio.gather(*tasks)
		#
		# 	for response, correct, lav in zip(results, correct_responses, look_ahead_values):
		# 		try:
		# 			if int(response) == int(correct):
		# 				print(f"Correct, look_ahead={lav}, num_paths={num_paths}, max_num_parents={max_num_parents}")
		# 			else:
		# 				print(f"Incorrect, response={response}, correct={correct} look_ahead={lav}, num_paths={num_paths}, max_num_parents={max_num_parents}")
		# 		except ValueError:
		# 			print(f"Response={response}, gave a value error, look_ahead={lav}")
		#
		# 	prompts = []
		# 	correct_responses = []
		# 	look_ahead_values = []



if __name__ == "__main__":
	asyncio.run(main())
	# num_paths = 2
	# max_num_parents = 3
	# look_ahead = 10
	#
	# # max_vertices = 3 * look_ahead
	# # max_input_size = 40
	#
	# txt, path = generate_graph_text(
	# 	max_input_size=max_num_parents * look_ahead * 4,
	# 	num_vertices=max_num_parents * look_ahead,
	# 	max_num_parents=max_num_parents,
	# 	lookahead=look_ahead,
	# 	num_paths=num_paths,
	# )

	# logic, logic_path = logic_paragraph_from_tokens(txt, path)
	# path = [node.id for node in path[0]]
	# print("\nRandom DAG:", txt, "\npath: ", path)
	# # print("\nLogic: ", logic, "\nLogic path: ", logic_path)

	# prompt = (f"{txt}\n\nAbove is a representation of a directed graph search problem, "
	# 		  f"where E A B represents an edge from A to B, and Q X Y represents starting from X and ending at Y,"
	# 		  f"find the shortest path. Respond with only the path in the form: V1 V2 V3 V4...")

	# client = OpenAI()


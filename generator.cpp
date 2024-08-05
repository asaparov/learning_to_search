#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <core/array.h>
#include <core/random.h>
#include <string>
#include <iostream>

using namespace core;

namespace py = pybind11;
using namespace py::literals;

static constexpr unsigned int RESERVED_INDICES[] = { 0 };
static const py::bool_ py_true(true);

template<typename T>
inline const T& choice(const T* items, unsigned int length) {
	return sample_uniform(items, length);
}

inline unsigned int randrange(unsigned int start, unsigned int end) {
	return start + sample_uniform(end - start);
}

inline unsigned int randrange(unsigned int end) {
	return sample_uniform(end);
}

struct node {
	unsigned int id;
	core::array<node*> children;
	core::array<node*> parents;

	node(unsigned int id) : id(id), children(8), parents(8) { }

	static inline void free(node& n) {
		core::free(n.children);
		core::free(n.parents);
	}
};

inline bool init(node& n, unsigned int id) {
	if (!array_init(n.children, 8)) {
		return false;
	} else if (!array_init(n.parents, 8)) {
		core::free(n.children);
		return false;
	}
	n.id = id;
	return true;
}

inline bool operator == (const node& first, const node& second) {
	return first.id == second.id;
}


#include <core/array.h>
#include <core/map.h>
#include <string>

using namespace core;


core::array<std::string> NAMES = core::array<std::string>(26);
core::array<std::string> NOUNS = core::array<std::string>(28);
core::array_map<std::string, std::string> CONNECTORS = core::array_map<std::string, std::string>(6);
core::array<std::string> ADDITIONAL_WORDS = core::array<std::string>(20);

void initialize_vocab() {


	NAMES.add("Alex");
	NAMES.add("Bob");
	NAMES.add("Charlie");
	NAMES.add("David");
	NAMES.add("Eve");
	NAMES.add("Fred");
	NAMES.add("Gina");
	NAMES.add("Hank");
	NAMES.add("Ivy");
	NAMES.add("Jack");
	NAMES.add("Kyle");
	NAMES.add("Lily");
	NAMES.add("Mia");
	NAMES.add("Nate");
	NAMES.add("Olivia");
	NAMES.add("Pam");
	NAMES.add("Quinn");
	NAMES.add("Ryan");
	NAMES.add("Sam");
	NAMES.add("Tara");
	NAMES.add("Uma");
	NAMES.add("Victor");
	NAMES.add("Wendy");
	NAMES.add("Xavier");
	NAMES.add("Yara");
	NAMES.add("Zara");
	
	NOUNS.add("qumpus");
	NOUNS.add("shumpus");
	NOUNS.add("grumpus");
	NOUNS.add("plumpus");
	NOUNS.add("clumpus");
	NOUNS.add("kumpus");
	NOUNS.add("sumpus");
	NOUNS.add("slumpus");
	NOUNS.add("umpus");
	NOUNS.add("flumpus");
	NOUNS.add("lumpus");
	NOUNS.add("rumpus");
	NOUNS.add("numpus");
	NOUNS.add("glumpus");
	NOUNS.add("mumpus");
	NOUNS.add("tumpus");
	NOUNS.add("humpus");
	NOUNS.add("bumpus");
	NOUNS.add("pumpus");
	NOUNS.add("xumpus");
	NOUNS.add("wumpus");
	NOUNS.add("jumpus");
	NOUNS.add("yumpus");
	NOUNS.add("zumpus");
	NOUNS.add("blumpus");
	NOUNS.add("dumpus");
	NOUNS.add("frumpus");
	NOUNS.add("vumpus");
	
	CONNECTORS.put("is a", "singular");
	CONNECTORS.put("has", "plural");
	CONNECTORS.put("wants", "plural");
	CONNECTORS.put("likes", "plural");
	CONNECTORS.put("cares for a", "singular");
	CONNECTORS.put("is friends with a", "singular");

	array<std::string> VOCAB = array<std::string>(150);

    VOCAB = NAMES;
    VOCAB.append(NOUNS.data, NOUNS.length);
    
    for (const auto& noun : NOUNS) {
        VOCAB.add(noun + "es");
    }

	ADDITIONAL_WORDS.add("a");
	ADDITIONAL_WORDS.add("is");
	ADDITIONAL_WORDS.add("has");
	ADDITIONAL_WORDS.add("wants");
	ADDITIONAL_WORDS.add("likes");
	ADDITIONAL_WORDS.add("cares");
	ADDITIONAL_WORDS.add("for");
	ADDITIONAL_WORDS.add("friends");
	ADDITIONAL_WORDS.add("with");
	ADDITIONAL_WORDS.add("then");
	ADDITIONAL_WORDS.add("Given");
	ADDITIONAL_WORDS.add("If");
	ADDITIONAL_WORDS.add("prove");
	ADDITIONAL_WORDS.add(".");
	ADDITIONAL_WORDS.add(" ");
	ADDITIONAL_WORDS.add(",");
	ADDITIONAL_WORDS.add("\n");
	ADDITIONAL_WORDS.add(":");
	ADDITIONAL_WORDS.add("Query");
	ADDITIONAL_WORDS.add("Prefix");
	ADDITIONAL_WORDS.add("Statements");
	ADDITIONAL_WORDS.add("[PAD]");
	ADDITIONAL_WORDS.add("[UNK]");
	ADDITIONAL_WORDS.add("[CLS]");
	ADDITIONAL_WORDS.add("[SEP]");
	ADDITIONAL_WORDS.add("[MASK]");

    VOCAB.append(ADDITIONAL_WORDS.data, ADDITIONAL_WORDS.length);
}

// Function to shuffle an array of strings
void shuffle_string_array(array<std::string>& arr) {
    if (arr.length <= 1) return;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int i = arr.length - 1; i > 0; --i) {
        std::uniform_int_distribution<> dis(0, i);
        int j = dis(gen);
        if (i != j) {
            std::swap(arr[i], arr[j]);
        }
    }
}


array<std::string> generate_atoms(unsigned int atom_count) {
    std::cout << "Debug: Entering generate_atoms with atom_count = " << atom_count << std::endl;
    array<std::string> atoms(atom_count);
    std::cout << "Debug: Initialized atoms array with capacity " << atom_count << std::endl;
    while (atoms.length < atom_count) {

		// print the connectors.keys array
		for (unsigned int i = 0; i < CONNECTORS.size; i++) {
			std::cout << "Debug: Connector: " << CONNECTORS.keys[i] << std::endl;
		}

        std::string connector = choice(CONNECTORS.keys, CONNECTORS.size);
        std::cout << "Debug: Selected connector: " << connector << std::endl;
        std::string predicate = choice(NOUNS.data, NOUNS.length);
        std::cout << "Debug: Selected predicate: " << predicate << std::endl;
        
        if (CONNECTORS.get(connector) == "plural") {
            predicate += "es";
            std::cout << "Debug: Pluralized predicate: " << predicate << std::endl;
        }
        
        std::string atom = choice(NAMES.data, NAMES.length) + " " + connector + " " + predicate + ".";
        std::cout << "Debug: Generated atom: " << atom << std::endl;
        if (!atoms.contains(atom)) {
            atoms.add(atom);
            std::cout << "Debug: Added atom to atoms array. Current length: " << atoms.length << std::endl;
        } else {
            std::cout << "Debug: Atom already exists, skipping" << std::endl;
        }
    }

    std::cout << "Debug: Shuffling atoms array" << std::endl;
    shuffle_string_array(atoms);

    std::cout << "Debug: Exiting generate_atoms, returning " << atoms.length << " atoms" << std::endl;
    return atoms;
}

array<std::string> generate_edge(unsigned int src, unsigned int dest, const array<std::string>& atom_map) {
    array<std::string> result(2);
    result.add("If " + atom_map[src].substr(0, atom_map[src].length() - 1) + ",");
    result.add("then " + atom_map[dest].substr(0, atom_map[dest].length() - 1) + ".");
    return result;
}

std::pair<std::string, std::string> map_tokens_to_natural_language(const array<int64_t>& tokens, int64_t output, unsigned int max_input_size, bool verbose = false) {
    const unsigned int QUERY_PREFIX_TOKEN = (max_input_size - 5) / 3 + 4;
    const unsigned int PADDING_TOKEN = (max_input_size - 5) / 3 + 3;
    const unsigned int EDGE_PREFIX_TOKEN = (max_input_size - 5) / 3 + 2;
    const unsigned int PATH_PREFIX_TOKEN = (max_input_size - 5) / 3 + 1;

    std::cout << "Debug: max_input_size = " << max_input_size << std::endl;
    std::cout << "Debug: QUERY_PREFIX_TOKEN = " << QUERY_PREFIX_TOKEN << std::endl;

    array<int64_t> unique_tokens(tokens.length);
    for (int64_t token : tokens) {
        if (token != QUERY_PREFIX_TOKEN && token != PADDING_TOKEN &&
            token != EDGE_PREFIX_TOKEN && token != PATH_PREFIX_TOKEN) {
			std::cout << "Debug: token = " << token << std::endl;
            if (!unique_tokens.contains(token))
                unique_tokens.add(token);
        }
    }

    std::cout << "Debug: unique_tokens.length = " << unique_tokens.length << std::endl;

    array<std::string> atoms = generate_atoms(unique_tokens.length);
    array_map<int64_t, std::string> token_to_atom(unique_tokens.length);
    for (size_t i = 0; i < unique_tokens.length; ++i) {
		std::cout << "Debug: unique_tokens[i] = " << unique_tokens[i] << " " << std::endl;
        token_to_atom.put(unique_tokens[i], atoms[i]);
    }

	std::cout << "Debug: token_to_atom.size = " << token_to_atom.size << std::endl;

    array<std::string> out_tokens(tokens.length);
    size_t i = 0;
    try {
        while (i < tokens.length) {
            std::cout << "Debug: Processing token " << i << ": " << tokens[i] << std::endl;
            if (tokens[i] == QUERY_PREFIX_TOKEN) {
                if (i + 2 >= tokens.length) {
                    std::cout << "Debug: Not enough tokens after QUERY_PREFIX_TOKEN" << std::endl;
                    break;
                }
                out_tokens.add("Given");
                out_tokens.add(token_to_atom.get(tokens[i+1]).substr(0, token_to_atom.get(tokens[i+1]).length() - 1) + ",");
                out_tokens.add("prove");
                out_tokens.add(token_to_atom.get(tokens[i+2]));
                i += 2;
            } else if (tokens[i] == EDGE_PREFIX_TOKEN) {
                if (i + 2 >= tokens.length) {
                    std::cout << "Debug: Not enough tokens after EDGE_PREFIX_TOKEN" << std::endl;
                    break;
                }
                array<std::string> edge = generate_edge(tokens[i+1], tokens[i+2], atoms);
                std::cout << "Debug: edge.length = " << edge.length << std::endl;
                out_tokens.append(edge.data, edge.length);
                i += 2;
            } else if (tokens[i] == PATH_PREFIX_TOKEN) {
                while (i + 1 < tokens.length) {
                    if (tokens[i+1] == PATH_PREFIX_TOKEN) {
                        // Do nothing
                    } else {
                        std::string atom = token_to_atom.get(tokens[i+1]);
						std::cout << "Debug: atom = " << atom << std::endl;
                    }
                    i++;
                }
            } else if (tokens[i] == PADDING_TOKEN) {
                // Do nothing
            }
            i++;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error mapping tokens to natural language: %s\n", e.what());
    }

    if (verbose) {
        for (size_t i = 0; i < tokens.length; ++i) {
            printf("%ld -> %s\n", tokens[i], (i < out_tokens.length ? out_tokens[i].c_str() : "N/A"));
        }
    }

    std::string full_out = "";
    for (const auto& token : out_tokens) {
        full_out += token + " ";
    }
    full_out += "\n" + token_to_atom.get(output);

    return std::make_pair(full_out, token_to_atom.get(output));
}

py::tuple map_tokens_to_natural_language_batched(const py::array_t<int64_t>& data, const py::array_t<int64_t>& output_tokens, unsigned int input_size, unsigned int TRANSFORMER_LENGTH, bool verbose = false) {
	
    auto data_unchecked = data.unchecked<2>();
    auto output_tokens_unchecked = output_tokens.unchecked<1>();
    size_t batch_size = data_unchecked.shape(0);

    std::cout << "Debug: batch_size = " << batch_size << std::endl;
    std::cout << "Debug: input_size = " << input_size << std::endl;

    array<std::string> all_tok(batch_size);
    array<std::string> all_out(batch_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        std::cout << "Debug: Processing batch " << i << std::endl;
        array<int64_t> tokens(input_size);
        for (size_t j = 0; j < input_size; ++j) {
            tokens.add(data_unchecked(i, j));
        }
        auto [tokens_str, output_str] = map_tokens_to_natural_language(tokens, output_tokens_unchecked(i), input_size, verbose);
        all_tok.add(tokens_str);
        all_out.add(output_str);
    }
    
    return py::make_tuple(all_tok, all_out);
}

/* computes the number of lookahead steps to find the answer */
unsigned int lookahead_depth(const node* vertex, const node* next_vertex, const node* goal)
{
	array<pair<const node*, const node*>> frontier(8);
	for (const node* v : vertex->children)
		frontier.add(make_pair(v, v));
	array<const node*> visited(16);
	visited.append(vertex->children.data, vertex->children.length);
	unsigned int lookahead = 0;
	while (frontier.length != 0) {
		bool frontier_is_next_vertex = true;
		for (pair<const node*, const node*>& entry : frontier) {
			if (entry.value != next_vertex) {
				frontier_is_next_vertex = false;
				break;
			}
		}
		if (frontier_is_next_vertex)
			return lookahead;

		lookahead++;
		array<pair<const node*, const node*>> new_frontier(8);
		for (const pair<const node*, const node*>& entry : frontier) {
			const node* v = entry.key;
			const node* branch = entry.value;

			if (v == goal)
				return lookahead;
			for (const node* child : v->children) {
				if (!visited.contains(child)) {
					new_frontier.add(make_pair(child, branch));
					visited.add(child);
				} else if (branch == next_vertex) {
					for (unsigned int i = 0; i < new_frontier.length; i++)
						if (new_frontier[i].key == child)
							new_frontier[i].value = branch;
				}
			}
		}
		core::swap(frontier, new_frontier);
	}
	return lookahead;
}

void get_descendants(const node& vertex, array<const node*>& descendants) {
	array<const node*> queue(8);
	array<const node*> visited(16);
	queue[0] = &vertex;
	queue.length = 1;
	while (queue.length != 0) {
		const node* current = queue.pop();
		visited.add(current);
		for (const node* child : current->children) {
			if (!descendants.contains(child))
				descendants.add(child);
			if (visited.contains(child))
				continue;
			queue.add(child);
		}
	}
}

bool has_cycles(array<node>& vertices) {
	for (const node& vertex : vertices) {
		array<const node*> descendants(8);
		get_descendants(vertex, descendants);
		if (descendants.contains(&vertex))
			return true;
	}
	return false;
}

bool generate_graph_with_lookahead(array<node>& vertices, node*& start, node*& end, unsigned int num_vertices, unsigned int max_num_parents, unsigned int max_vertex_id, unsigned int lookahead, unsigned int num_paths)
{
	num_vertices = std::max(std::max(2u, num_vertices), 1 + num_paths * lookahead);

	if (!vertices.ensure_capacity(num_vertices))
		return false;
	for (unsigned int i = 0; i < num_vertices; i++)
		if (!init(vertices[i], i)) return false;
	vertices.length = num_vertices;

	vertices[1].parents.add(&vertices[0]);
	vertices[0].children.add(&vertices[1]);
	for (unsigned int i = 1; i < lookahead; i++) {
		vertices[1 + i].parents.add(&vertices[i]);
		vertices[i].children.add(&vertices[1 + i]);
	}
	unsigned int index;
	if (lookahead == 0) {
		index = 2;
	} else {
		index = 1 + lookahead;
		for (unsigned int j = 0; j < num_paths - 1; j++) {
			vertices[index].parents.add(&vertices[0]);
			vertices[0].children.add(&vertices[index]);
			index++;
			unsigned int other_branch_length = lookahead + randrange(std::min(2u, num_vertices - index - (num_paths - j - 1) * lookahead + 2));
			for (unsigned int i = 1; i < other_branch_length; i++) {
				vertices[index].parents.add(&vertices[index - 1]);
				vertices[index - 1].children.add(&vertices[index]);
				index++;
			}
		}
	}

	unsigned int num_prefix_vertices = randrange(num_vertices - index + 1);
	node* prev_vertex = &vertices[0];
	for (unsigned int i = 0; i < num_prefix_vertices; i++) {
		vertices[index].children.add(prev_vertex);
		prev_vertex->parents.add(&vertices[index]);
		prev_vertex = &vertices[index];
		index++;
	}

	start = &vertices[0];
	end = &vertices[std::max(1u, lookahead)];

	/* sample some parent/ancestor vertices */
	constexpr float ALPHA = 0.5f;
	unsigned int* in_degrees = (unsigned int*) calloc(num_vertices, sizeof(unsigned int));
	unsigned int* out_degrees = (unsigned int*) calloc(num_vertices, sizeof(unsigned int));
	if (in_degrees == nullptr || out_degrees == nullptr) {
		if (in_degrees != nullptr) free(in_degrees);
		return false;
	}
	for (unsigned int i = 0; i < num_vertices; i++) {
		in_degrees[i] = vertices[i].parents.length;
		out_degrees[i] = vertices[i].children.length;
	}
	for (unsigned int i = index; i < num_vertices; i++) {
		/* sample the number of child and parent vertices */
		unsigned int num_children = randrange(0, max_num_parents);
		unsigned int num_parents = randrange(num_children == 0 ? 1 : 0, max_num_parents);
		num_children = std::min(num_children, i);
		num_parents = std::min(num_parents, i);

		/* sample the children of this new node */
		array<float> probabilities(index);
		float total_probability = 0.0f;
		for (unsigned int j = 0; j < index; j++) {
			probabilities[j] = ALPHA + in_degrees[j];
			total_probability += probabilities[j];
		}
		probabilities.length = index;

		array<unsigned int> sampled_children(std::max(1u, num_children));
		for (unsigned int j = 0; j < num_children; j++) {
			unsigned int u = sample_categorical(probabilities.data, total_probability, probabilities.length);
			sampled_children.add(u);
			total_probability -= probabilities[u];
			probabilities[u] = 0.0f;
		}

		for (unsigned int child_id : sampled_children) {
			vertices[index].children.add(&vertices[child_id]);
			vertices[child_id].parents.add(&vertices[index]);
			in_degrees[child_id] += 1;
		}

		/* sample the parents of this new node */
		total_probability = 0.0f;
		for (unsigned int j = 0; j < index; j++) {
			probabilities[j] = ALPHA + out_degrees[j];
			total_probability += probabilities[j];
		}

		/* to avoid creating a cycle, we have to remove any descendants from the possible parents */
		array<const node*> descendants(8);
		get_descendants(vertices[index], descendants);
		for (const node* descendant : descendants) {
			total_probability -= probabilities[descendant->id];
			probabilities[descendant->id] = 0.0f;
		}
		if (total_probability != 0.0f) {
			num_parents = std::min(num_parents,  index - (unsigned int) descendants.length);

			array<unsigned int> sampled_parents(std::max(1u, num_parents));
			for (unsigned int j = 0; j < num_parents; j++) {
				unsigned int u = sample_categorical(probabilities.data, total_probability, probabilities.length);
				sampled_parents.add(u);
				total_probability -= probabilities[u];
				probabilities[u] = 0.0f;
			}

			for (unsigned int parent_id : sampled_parents) {
				vertices[parent_id].children.add(&vertices[index]);
				vertices[index].parents.add(&vertices[parent_id]);
				out_degrees[parent_id] += 1;
			}
		}
		index += 1;
	}
	free(in_degrees);
	free(out_degrees);

	/* remove any correlation between graph topology and vertex IDs by shuffling the vertices */
	unsigned int* new_indices = (unsigned int*) alloca(sizeof(unsigned int) * (max_vertex_id + 1));
	for (unsigned int i = 0; i < max_vertex_id + 1; i++) new_indices[i] = i;
	shuffle(new_indices, max_vertex_id + 1);
	unsigned int src_index = 0;
	for (unsigned int i = 0; i < vertices.length; i++) {
		bool is_reserved = false;
		for (unsigned int j = 0; j < array_length(RESERVED_INDICES); j++) {
			if (new_indices[src_index] == RESERVED_INDICES[j]) {
				is_reserved = true;
				break;
			}
		}
		if (is_reserved)
			src_index++;
		vertices[i].id = new_indices[src_index];
		src_index++;
	}
	return true;
}

bool generate_example(array<node>& vertices, node*& start, node*& end, array<array<node*>>& paths, unsigned int num_vertices, unsigned int max_num_parents, unsigned int max_vertex_id, bool get_shortest_paths, unsigned int lookahead, unsigned int num_paths)
{
	if (!generate_graph_with_lookahead(vertices, start, end, num_vertices, max_num_parents, max_vertex_id, lookahead, num_paths))
		return false;

	/* find the shortest paths from `start` to `end` */
	array<pair<node*, unsigned int>> queue(16);
	queue[0].key = start;
	queue[0].value = 0;
	queue.length = 1;
	array_map<node*, array_map<node*, unsigned int>> reverse_pointers(16);
	while (queue.length != 0) {
		pair<node*, unsigned int> item = queue.pop();
		node* current = item.key;
		unsigned int distance = item.value;

		for (node* child : current->children) {
			if (!reverse_pointers.ensure_capacity(reverse_pointers.size + 1)) {
				for (const auto& entry : reverse_pointers) core::free(entry.value);
				return false;
			}
			bool contains;
			array_map<node*, unsigned int>& value = reverse_pointers.get(child, contains);
			if (!contains) {
				if (!array_map_init(value, 4)) {
					for (const auto& entry : reverse_pointers) core::free(entry.value);
					return false;
				}
				reverse_pointers.keys[reverse_pointers.size++] = child;
			}

			if (!value.ensure_capacity(value.size + 1)) {
				for (const auto& entry : reverse_pointers) core::free(entry.value);
				return false;
			}
			unsigned int& distance_value = value.get(current, contains);
			if (!contains) {
				distance_value = distance + 1;
				value.keys[value.size++] = current;
			} else if (distance_value > distance + 1) {
				distance_value = distance + 1;
			} else {
				continue;
			}

			bool found_child = false;
			for (unsigned int j = 0; j < queue.length; j++) {
				if (queue[j].key == child) {
					queue[j].value = std::min(queue[j].value, distance + 1);
					found_child = true;
					break;
				}
			}
			if (!found_child)
				queue.add(make_pair(child, distance + 1));
		}
	}

	if (!reverse_pointers.contains(end)) {
		for (const auto& entry : reverse_pointers) core::free(entry.value);
		return false;
	}

	array_map<const node*, array<node*>> forward_pointers(16);
	array<node*> dist_queue(16);
	dist_queue[0] = end;
	dist_queue.length = 1;
	while (dist_queue.length != 0) {
		node* current = dist_queue.pop();
		if (current == start)
			continue;

		array<node*> prev_nodes(8);
		const array_map<node*, unsigned int>& value = reverse_pointers.get(current);
		if (get_shortest_paths) {
			unsigned int min_distance = value.values[0];
			for (unsigned int i = 1; i < value.size; i++)
				if (value.values[i] < min_distance) min_distance = value.values[i];

			for (unsigned int i = 0; i < value.size; i++)
				if (value.values[i] == min_distance) prev_nodes.add(value.keys[i]);
		} else {
			prev_nodes.append(value.keys, value.size);
		}
		for (const node* prev : prev_nodes) {
			if (!forward_pointers.ensure_capacity(forward_pointers.size + 1)) {
				for (const auto& entry : reverse_pointers) core::free(entry.value);
				for (const auto& entry : forward_pointers) core::free(entry.value);
				return false;
			}
			bool contains;
			array<node*>& fptrs = forward_pointers.get(prev, contains);
			if (!contains) {
				if (!array_init(fptrs, 4)) {
					for (const auto& entry : reverse_pointers) core::free(entry.value);
					for (const auto& entry : forward_pointers) core::free(entry.value);
					return false;
				}
				forward_pointers.keys[forward_pointers.size++] = prev;
			}
			fptrs.add(current);
		}
		dist_queue.append(prev_nodes.data, prev_nodes.length);
	}

	/* free `reverse_pointers` */
	for (const auto& entry : reverse_pointers) core::free(entry.value);

	/* construct the shortest paths from the forward pointers */
	array<array<node*>> path_queue(8);
	if (!array_init(path_queue[0], 1)) {
		for (const auto& entry : forward_pointers) core::free(entry.value);
		return false;
	}
	path_queue[0].add(start);
	path_queue.length = 1;
	while (path_queue.length != 0) {
		array<node*>& partial_path = *((array<node*>*) alloca(sizeof(array<node*>)));
		core::move(path_queue.last(), partial_path);
		path_queue.length--;

		if (partial_path.last() == end) {
			if (!paths.ensure_capacity(paths.length + 1)) {
				for (const auto& entry : forward_pointers) core::free(entry.value);
				for (auto& p : path_queue) core::free(p);
				core::free(partial_path);
				return false;
			}
			core::move(partial_path, paths[paths.length++]);
			if (paths.length > 64) {
				for (const auto& entry : forward_pointers) core::free(entry.value);
				for (auto& p : path_queue) core::free(p);
				return false;
			}
			continue;
		}
		for (node* next : forward_pointers.get(partial_path.last())) {
			if (!path_queue.ensure_capacity(path_queue.length + 1)
			 || !array_init(path_queue[path_queue.length], partial_path.length + 1))
			{
				for (const auto& entry : forward_pointers) core::free(entry.value);
				for (auto& p : path_queue) core::free(p);
				core::free(partial_path);
				return false;
			}
			path_queue.length++;
			path_queue.last().append(partial_path.data, partial_path.length);
			path_queue.last().add(next);
		}
		core::free(partial_path);
	}

	for (const auto& entry : forward_pointers) core::free(entry.value);
	return true;
}

bool has_path(const node* start, const node* end)
{
	array<const node*> stack(8);
	stack[0] = start;
	stack.length = 1;
	array<const node*> visited(16);
	while (stack.length != 0) {
		const node* v = stack.pop();
		if (v == end)
			return true;
		for (node* child : v->children) {
			if (!visited.contains(child)) {
				visited.add(child);
				stack.add(child);
			}
		}
	}
	return false;
}

py::tuple generate_training_set(const unsigned int max_input_size, const uint64_t dataset_size, const unsigned int max_lookahead, const unsigned int max_edges, const py::object& reserved_inputs, const int distance_from_start, const bool nl, const bool quiet=false)
{
	const unsigned int QUERY_PREFIX_TOKEN = (max_input_size-5) / 3 + 4;
	const unsigned int PADDING_TOKEN = (max_input_size-5) / 3 + 3;
	const unsigned int EDGE_PREFIX_TOKEN = (max_input_size-5) / 3 + 2;
	const unsigned int PATH_PREFIX_TOKEN = (max_input_size-5) / 3 + 1;

	unsigned int num_generated = 0;
	unsigned int num_collisions = 0;
	unsigned int ntokens = (max_input_size - 5) / 3 + 5;
	size_t input_shape[2]{dataset_size, max_input_size};
	size_t output_shape[2]{dataset_size, ntokens};
	size_t label_shape[1]{dataset_size};
	py::array_t<int64_t, py::array::c_style> inputs(input_shape);
	py::array_t<float, py::array::c_style> outputs(output_shape);
	py::array_t<int64_t, py::array::c_style> labels(label_shape);
	auto inputs_mem = inputs.mutable_unchecked<2>();
	auto outputs_mem = outputs.mutable_unchecked<2>();
	auto labels_mem = labels.mutable_unchecked<1>();
	unsigned int* lookahead_step_histogram = (unsigned int*) alloca(sizeof(unsigned int) * max_input_size);
	unsigned int* path_length_histogram = (unsigned int*) alloca(sizeof(unsigned int) * max_input_size);
	for (unsigned int i = 0; i < max_input_size; i++) {
		lookahead_step_histogram[i] = 0;
		path_length_histogram[i] = 0;
	}
	float* MAX_FREQS_PER_BUCKET = (float*) alloca(sizeof(float) * max_input_size);
	for (unsigned int i = 0; i < max_lookahead + 1; i++)
		MAX_FREQS_PER_BUCKET[i] = 1.0 / (max_lookahead+1);
	for (unsigned int i = max_lookahead + 1; i < max_input_size; i++)
		MAX_FREQS_PER_BUCKET[i] = 0.0;
	MAX_FREQS_PER_BUCKET[max_lookahead] += 0.05;

	unsigned int* potential_lookaheads = (unsigned int*) alloca(sizeof(unsigned int) * (max_lookahead + 1));
	unsigned int potential_lookahead_count = 0;
	while (num_generated < dataset_size) {
		array<node> g(32);
		node* start; node* end;
		array<array<node*>> paths(8);
		while (true) {
			potential_lookahead_count = 0;
			for (unsigned int i = 0; i < max_lookahead + 1; i++)
				if (num_generated == 0 || lookahead_step_histogram[i] / num_generated < MAX_FREQS_PER_BUCKET[i])
					potential_lookaheads[potential_lookahead_count++] = i;
			unsigned int lookahead = choice(potential_lookaheads, potential_lookahead_count);

			unsigned int num_paths;
			if (lookahead == 0) {
				num_paths = randrange(1, 3);
			} else {
				unsigned int max_num_paths = (max_edges - 1) / lookahead;
				num_paths = randrange(2, max_num_paths + 1);
			}

			unsigned int num_vertices = std::min(std::min(lookahead * num_paths + 1 + randrange(0, 6), (max_input_size-5) / 3), max_edges + 1);
			if (!generate_example(g, start, end, paths, num_vertices, 4, (max_input_size - 5) / 3, true, lookahead, num_paths)) {
				for (node& n : g) core::free(n);
				for (array<node*>& a : paths) core::free(a);
				g.length = 0; paths.length = 0;
				continue;
			}
			unsigned int shortest_path_length = paths[0].length;
			for (unsigned int i = 1; i < paths.length; i++)
				if (paths[i].length < shortest_path_length)
					shortest_path_length = paths[i].length;
			if (shortest_path_length > 1)
				break;
			for (node& n : g) core::free(n);
			for (array<node*>& a : paths) core::free(a);
			g.length = 0; paths.length = 0;
		}

		array<pair<unsigned int, unsigned int>> edges(8);
		for (node& vertex : g)
			for (node* child : vertex.children)
				edges.add(make_pair(vertex.id, child->id));
		if (edges.length > max_edges || edges.length * 3 + 4 > max_input_size) {
			for (node& n : g) core::free(n);
			for (array<node*>& a : paths) core::free(a);
			g.length = 0; paths.length = 0;
			continue;
		}
		shuffle(edges);

		array<unsigned int> prefix(max_input_size);
		for (auto& entry : edges) {
			prefix[prefix.length++] = EDGE_PREFIX_TOKEN;
			prefix[prefix.length++] = entry.key;
			prefix[prefix.length++] = entry.value;
		}
		prefix[prefix.length++] = QUERY_PREFIX_TOKEN;
		prefix[prefix.length++] = start->id;
		prefix[prefix.length++] = end->id;
		prefix[prefix.length++] = PATH_PREFIX_TOKEN;

		for (const array<node*>& path : paths) {
			if (path.length == 1)
				continue;
			for (unsigned int j = 1; j < path.length; j++) {
				if (distance_from_start != -1 && j != (unsigned int) distance_from_start)
					continue;
				array<unsigned int> example(prefix.length + j);
				for (unsigned int i = 0; i < prefix.length; i++)
					example[i] = prefix[i];
				for (unsigned int i = 0; i < j; i++)
					example[prefix.length + i] = path[i]->id;
				example.length = prefix.length + j;
				if (example.length > max_input_size)
					continue;

				unsigned int lookahead_steps = lookahead_depth(path[j-1], path[j], end);
				array<node*> useful_steps(8);
				for (node* v : path[j-1]->children)
					if (has_path(v, end)) useful_steps.add(v);

				/* check if this input is reserved */
				py::object contains = reserved_inputs.attr("__contains__");
				py::tuple example_tuple(example.length);
				for (unsigned int i = 0; i < example.length; i++)
					example_tuple[i] = example[i];
				if (contains(example_tuple).is(py_true)) {
					num_collisions += 1;
					continue;
				}

				if (num_generated != 0 && lookahead_step_histogram[lookahead_steps] / num_generated >= MAX_FREQS_PER_BUCKET[lookahead_steps])
					continue;
				lookahead_step_histogram[lookahead_steps] += 1;
				path_length_histogram[j] += 1;

				for (unsigned int i = 0; i < max_input_size - example.length; i++)
					inputs_mem(num_generated, i) = PADDING_TOKEN;
				for (unsigned int i = 0; i < example.length; i++)
					inputs_mem(num_generated, max_input_size - example.length + i) = example[i];
				for (unsigned int i = 0; i < ntokens; i++)
					outputs_mem(num_generated, i) = 0.0f;
				for (unsigned int i = 0; i < useful_steps.length; i++)
					outputs_mem(num_generated, useful_steps[i]->id) = 1.0f;
				labels_mem(num_generated) = choice(useful_steps.data, useful_steps.length)->id;
				num_generated++;
				if (num_generated == dataset_size)
					break;
			}
			if (num_generated == dataset_size)
				break;
		}

		if (!quiet && num_generated > 0 && (num_generated % 1000 == 0 || num_generated >= dataset_size)) {
			printf("%d examples generated.\n", num_generated);

			printf("Lookahead steps histogram:\n");
			printf("[");
			bool first = true;
			for (unsigned int i = 0; i < max_input_size; i++) {
				if (lookahead_step_histogram[i] == 0)
					continue;
				if (!first) printf(", ");
				printf("%d:%.2f", i, (float) lookahead_step_histogram[i] / num_generated + 1e-9);
				first = false;
			}
			printf("]\n");

			printf("Path length histogram:\n");
			printf("[");
			first = true;
			for (unsigned int i = 0; i < max_input_size; i++) {
				if (path_length_histogram[i] == 0)
					continue;
				if (!first) printf(", ");
				printf("%d:%.2f", i, (float) path_length_histogram[i] / num_generated + 1e-9);
				first = false;
			}
			printf("]\n");
			fflush(stdout);
		}

		for (node& n : g) core::free(n);
		for (array<node*>& a : paths) core::free(a);
		g.length = 0; paths.length = 0;
		continue;
	}

    if (true) {
        // auto [new_inputs, new_labels] = map_tokens_to_natural_language_batched(inputs, labels, max_input_size, 192);
		py::tuple result = map_tokens_to_natural_language_batched(inputs, labels, max_input_size, 192);
        py::list new_inputs = result[0].cast<py::list>();
        py::list new_labels = result[1].cast<py::list>();

		return py::make_tuple(new_inputs, outputs, new_labels, num_collisions);
    }
	else
		return py::make_tuple(inputs, outputs, labels, num_collisions);
}





py::tuple generate_reachable_training_set(const unsigned int max_input_size, const uint64_t dataset_size, const unsigned int lookahead, const unsigned int max_edges, const py::object& reserved_inputs, const int distance_from_start, const int reachable_distance, const unsigned int start_vertex_index, const bool exclude_start_vertex)
{
	const unsigned int QUERY_PREFIX_TOKEN = (max_input_size-5) / 3 + 4;
	const unsigned int PADDING_TOKEN = (max_input_size-5) / 3 + 3;
	const unsigned int EDGE_PREFIX_TOKEN = (max_input_size-5) / 3 + 2;
	const unsigned int PATH_PREFIX_TOKEN = (max_input_size-5) / 3 + 1;

	unsigned int num_generated = 0;
	unsigned int num_collisions = 0;
	size_t input_shape[2]{dataset_size, max_input_size};
	size_t output_shape[2]{dataset_size, max_input_size};
	py::array_t<int64_t, py::array::c_style> inputs(input_shape);
	py::array_t<float, py::array::c_style> outputs(output_shape);
	auto inputs_mem = inputs.mutable_unchecked<2>();
	auto outputs_mem = outputs.mutable_unchecked<2>();
	py::list valid_outputs;

	unsigned int max_vertex_id = (max_input_size - 5) / 3;
	while (num_generated < dataset_size) {
		array<node> g(32);
		node* start; node* end;
		array<array<node*>> paths(8);
		while (true) {
			unsigned int num_paths;
			if (lookahead == 0) {
				num_paths = randrange(1, 3);
			} else {
				unsigned int max_num_paths = (max_edges - 1) / lookahead;
				num_paths = randrange(2, max_num_paths + 1);
			}

			unsigned int num_vertices = std::min(std::min(lookahead * num_paths + 1 + randrange(0, 6), (max_input_size-5) / 3), max_edges + 1);
			if (!generate_example(g, start, end, paths, num_vertices, 4, max_vertex_id, true, lookahead, num_paths)) {
				for (node& n : g) core::free(n);
				for (array<node*>& a : paths) core::free(a);
				g.length = 0; paths.length = 0;
				continue;
			}
			unsigned int shortest_path_length = paths[0].length;
			for (unsigned int i = 1; i < paths.length; i++)
				if (paths[i].length < shortest_path_length)
					shortest_path_length = paths[i].length;
			if (shortest_path_length > 1)
				break;
			for (node& n : g) core::free(n);
			for (array<node*>& a : paths) core::free(a);
			g.length = 0; paths.length = 0;
		}

		array<pair<unsigned int, unsigned int>> edges(8);
		for (node& vertex : g)
			for (node* child : vertex.children)
				edges.add(make_pair(vertex.id, child->id));
		if (edges.length > max_edges || edges.length * 3 + 4 > max_input_size) {
			for (node& n : g) core::free(n);
			for (array<node*>& a : paths) core::free(a);
			g.length = 0; paths.length = 0;
			continue;
		}
		shuffle(edges);

		array<unsigned int> prefix(max_input_size);
		for (auto& entry : edges) {
			prefix[prefix.length++] = EDGE_PREFIX_TOKEN;
			prefix[prefix.length++] = entry.key;
			prefix[prefix.length++] = entry.value;
		}
		prefix[prefix.length++] = QUERY_PREFIX_TOKEN;
		prefix[prefix.length++] = start->id;
		prefix[prefix.length++] = end->id;
		prefix[prefix.length++] = PATH_PREFIX_TOKEN;

		for (const array<node*>& path : paths) {
			if (path.length == 1)
				continue;
			for (unsigned int j = 1; j < path.length; j++) {
				if (distance_from_start != -1 && j != (unsigned int) distance_from_start)
					continue;
				array<unsigned int> example(prefix.length + j);
				for (unsigned int i = 0; i < prefix.length; i++)
					example[i] = prefix[i];
				for (unsigned int i = 0; i < j; i++)
					example[prefix.length + i] = path[i]->id;
				example.length = prefix.length + j;
				if (example.length > max_input_size)
					continue;

				/* compute the set of reachable vertices */
				node** vertex_id_map = (node**) calloc(max_vertex_id + 1, sizeof(node*));
				for (unsigned int i = 0; i < g.length; i++)
					vertex_id_map[g[i].id] = &g[i];
				array<unsigned int> reachable(16);
				array<pair<unsigned int, unsigned int>> stack(16);
				unsigned int start_vertex;
				if (example.length < start_vertex_index)
					start_vertex = start->id;
				else start_vertex = example[example.length - start_vertex_index];
				stack.add(make_pair(start_vertex, 0u));
				while (stack.length != 0) {
					pair<unsigned int, unsigned int> entry = stack.pop();
					unsigned int current_vertex = entry.key;
					unsigned int current_distance = entry.value;
					if (!reachable.contains(current_vertex))
						reachable.add(current_vertex);
					if (reachable_distance > 0 && current_distance + 1 <= (unsigned int) reachable_distance) {
						for (node* child : vertex_id_map[current_vertex]->children)
							stack.add(make_pair(child->id, current_distance + 1));
					} else if (reachable_distance < 0 && current_distance + 1 <= (unsigned int) -reachable_distance) {
						for (node* parent : vertex_id_map[current_vertex]->parents)
							stack.add(make_pair(parent->id, current_distance + 1));
					}
				}
				if (exclude_start_vertex)
					reachable.remove(reachable.index_of(start_vertex));

				array<node*> useful_steps(8);
				for (node* v : path[j-1]->children)
					if (has_path(v, end)) useful_steps.add(v);

				/* check if this input is reserved */
				py::object contains = reserved_inputs.attr("__contains__");
				py::tuple example_tuple(example.length);
				for (unsigned int i = 0; i < example.length; i++)
					example_tuple[i] = example[i];
				if (contains(example_tuple).is(py_true)) {
					num_collisions += 1;
					continue;
				}

				for (unsigned int i = 0; i < max_input_size - example.length; i++)
					inputs_mem(num_generated, i) = PADDING_TOKEN;
				for (unsigned int i = 0; i < example.length; i++)
					inputs_mem(num_generated, max_input_size - example.length + i) = example[i];
				for (unsigned int i = 0; i < max_input_size - example.length; i++)
					outputs_mem(num_generated, i) = 0;
				for (unsigned int i = 0; i < example.length; i++)
					outputs_mem(num_generated, max_input_size - example.length + i) = reachable.contains(example[i]) ? 1 : 0;
				num_generated++;
				if (num_generated == dataset_size)
					break;
			}
			if (num_generated == dataset_size)
				break;
		}

		for (node& n : g) core::free(n);
		for (array<node*>& a : paths) core::free(a);
		g.length = 0; paths.length = 0;
		continue;
	}

	return py::make_tuple(inputs, outputs, num_collisions);
}

bool generate_dfs_example(array<node>& vertices, const node*& start, const node*& end, array<const node*>& path, unsigned int num_vertices, unsigned int max_num_parents, unsigned int max_vertex_id)
{
	if (!vertices.ensure_capacity(num_vertices))
		return false;
	for (unsigned int i = 0; i < num_vertices; i++)
		if (!init(vertices[i], i)) return false;
	vertices.length = num_vertices;

	/* sample some parent/ancestor vertices */
	constexpr float ALPHA = 1.0f;
	unsigned int* out_degrees = (unsigned int*) calloc(num_vertices, sizeof(unsigned int));
	if (out_degrees == nullptr)
		return false;
	for (unsigned int i = 1; i < num_vertices; i++) {
		/* sample the number of child and parent vertices */
		unsigned int num_parents = randrange(1, max_num_parents);
		num_parents = std::min(num_parents, i);

		/* sample the parents of this new node */
		float total_probability = 0.0f;
		array<float> probabilities(i);
		for (unsigned int j = 0; j < i; j++) {
			probabilities[j] = ALPHA + out_degrees[j];
			total_probability += probabilities[j];
		}
		probabilities.length = i;

		array<unsigned int> sampled_parents(std::max(1u, num_parents));
		for (unsigned int j = 0; j < num_parents; j++) {
			unsigned int u = sample_categorical(probabilities.data, total_probability, probabilities.length);
			sampled_parents.add(u);
			total_probability -= probabilities[u];
			probabilities[u] = 0.0f;
		}

		for (unsigned int parent_id : sampled_parents) {
			vertices[parent_id].children.add(&vertices[i]);
			vertices[i].parents.add(&vertices[parent_id]);
			out_degrees[parent_id] += 1;
		}
	}
	free(out_degrees);

	/* remove any correlation between graph topology and vertex IDs by shuffling the vertices */
	unsigned int* new_indices = (unsigned int*) alloca(sizeof(unsigned int) * (max_vertex_id + 1));
	for (unsigned int i = 0; i < max_vertex_id + 1; i++) new_indices[i] = i;
	shuffle(new_indices, max_vertex_id + 1);
	unsigned int src_index = 0;
	for (unsigned int i = 0; i < vertices.length; i++) {
		bool is_reserved = false;
		for (unsigned int j = 0; j < array_length(RESERVED_INDICES); j++) {
			if (new_indices[src_index] == RESERVED_INDICES[j]) {
				is_reserved = true;
				break;
			}
		}
		if (is_reserved)
			src_index++;
		vertices[i].id = new_indices[src_index];
		src_index++;
	}

	while (true) {
		/* select a start and goal vertex uniformly at random */
		start = &choice(vertices.data, vertices.length);
		end = &choice(vertices.data, vertices.length);
		if (start == end) continue;

		/* perform DFS from the start vertex */
		array<const node*> queue(8);
		queue[0] = start;
		queue.length = 1;
		bool found_goal = false;
		while (queue.length != 0) {
			const node* current = queue.pop();
			path.add(current);

			if (current->children.contains(end)) {
				found_goal = true;
				path.add(end);
				break;
			}

			for (const node* child : current->children) {
				if (path.contains(child)) continue;
				queue.add(child);
			}
		}

		/* check if the goal vertex is reachable from the start vertex */
		if (found_goal)
			break;
		path.clear();
	}

	return true;
}

py::tuple generate_dfs_training_set(const unsigned int max_input_size, const uint64_t dataset_size, const py::object& reserved_inputs, const int requested_backtrack, const bool quiet=false)
{
	const unsigned int QUERY_PREFIX_TOKEN = (max_input_size-5) / 3 + 4;
	const unsigned int PADDING_TOKEN = (max_input_size-5) / 3 + 3;
	const unsigned int EDGE_PREFIX_TOKEN = (max_input_size-5) / 3 + 2;
	const unsigned int PATH_PREFIX_TOKEN = (max_input_size-5) / 3 + 1;

	unsigned int longest_path_length = (max_input_size - 4) / 4;

	unsigned int num_generated = 0;
	unsigned int num_collisions = 0;
	unsigned int ntokens = (max_input_size - 5) / 3 + 5;
	size_t input_shape[2]{dataset_size, max_input_size};
	size_t output_shape[2]{dataset_size, ntokens};
	size_t label_shape[1]{dataset_size};
	py::array_t<int64_t, py::array::c_style> inputs(input_shape);
	py::array_t<float, py::array::c_style> outputs(output_shape);
	py::array_t<int64_t, py::array::c_style> labels(label_shape);
	auto inputs_mem = inputs.mutable_unchecked<2>();
	auto outputs_mem = outputs.mutable_unchecked<2>();
	auto labels_mem = labels.mutable_unchecked<1>();

	unsigned int* backtrack_distance_histogram = (unsigned int*) alloca(sizeof(unsigned int) * max_input_size);
	for (unsigned int i = 0; i < max_input_size; i++)
		backtrack_distance_histogram[i] = 0;

	array<const node*> path(32);
	while (num_generated < dataset_size) {
		array<node> g(32);
		const node* start; const node* end;
		while (true) {
			unsigned int num_vertices = std::max(2u, randrange(longest_path_length + 1));
			if (requested_backtrack != -1)
				num_vertices = std::max((unsigned int) requested_backtrack + 3, num_vertices);
			if (!generate_dfs_example(g, start, end, path, num_vertices, max_input_size / 24 + 1, (max_input_size - 5) / 3)) {
				for (node& n : g) core::free(n);
				g.length = 0; path.length = 0;
				continue;
			}
			break;
		}

		array<pair<unsigned int, unsigned int>> edges(8);
		for (node& vertex : g)
			for (node* child : vertex.children)
				edges.add(make_pair(vertex.id, child->id));
		if (edges.length > longest_path_length) {
			for (node& n : g) core::free(n);
			g.length = 0; path.length = 0;
			continue;
		}
		shuffle(edges);

		array<unsigned int> prefix(max_input_size);
		for (auto& entry : edges) {
			prefix[prefix.length++] = EDGE_PREFIX_TOKEN;
			prefix[prefix.length++] = entry.key;
			prefix[prefix.length++] = entry.value;
		}
		prefix[prefix.length++] = QUERY_PREFIX_TOKEN;
		prefix[prefix.length++] = start->id;
		prefix[prefix.length++] = end->id;
		prefix[prefix.length++] = PATH_PREFIX_TOKEN;

		/* randomly select a vertex in the DFS trace */
		unsigned int index = randrange(path.length - 1);
		path.length = index + 1;

		array<const node*> unvisited(4);
		unsigned int backtrack_distance = max_input_size - 1;
		for (unsigned int j = index + 1; j > 0; j--) {
			for (const node* child : path[j-1]->children) {
				if (!path.contains(child))
					unvisited.add(child);
			}
			if (unvisited.length != 0) {
				backtrack_distance = index + 1 - j;
				break;
			}
		}

		if (requested_backtrack != -1 && (unsigned int) requested_backtrack != backtrack_distance) {
			for (node& n : g) core::free(n);
			g.length = 0; path.length = 0;
			continue;
		}

		while (prefix.length < max_input_size - path.length)
			prefix[prefix.length++] = PATH_PREFIX_TOKEN;
		for (unsigned int j = 0; j < path.length; j++)
			prefix[prefix.length++] = path[j]->id;

		/* check if this input is reserved */
		py::object contains = reserved_inputs.attr("__contains__");
		py::tuple example_tuple(prefix.length);
		for (unsigned int i = 0; i < prefix.length; i++)
			example_tuple[i] = prefix[i];
		if (contains(example_tuple).is(py_true)) {
			for (node& n : g) core::free(n);
			g.length = 0; path.length = 0;
			num_collisions += 1;
			continue;
		}

		backtrack_distance_histogram[backtrack_distance]++;

		for (unsigned int i = 0; i < max_input_size - prefix.length; i++)
			inputs_mem(num_generated, i) = PADDING_TOKEN;
		for (unsigned int i = 0; i < prefix.length; i++)
			inputs_mem(num_generated, max_input_size - prefix.length + i) = prefix[i];
		for (unsigned int i = 0; i < ntokens; i++)
			outputs_mem(num_generated, i) = 0.0f;
		for (unsigned int i = 0; i < unvisited.length; i++)
			outputs_mem(num_generated, unvisited[i]->id) = 1.0f;
		labels_mem(num_generated) = path[index+1]->id;
		num_generated++;

		if (!quiet && num_generated > 0 && (num_generated % 1000 == 0 || num_generated >= dataset_size)) {
			printf("%d examples generated.\n", num_generated);
			fflush(stdout);

			printf("Backtrack distance histogram:\n");
			printf("[");
			bool first = true;
			for (unsigned int i = 0; i < max_input_size; i++) {
				if (backtrack_distance_histogram[i] == 0)
					continue;
				if (!first) printf(", ");
				printf("%d:%.2f", i, (float) backtrack_distance_histogram[i] / num_generated + 1e-9);
				first = false;
			}
			printf("]\n");
		}

		for (node& n : g) core::free(n);
		g.length = 0; path.length = 0;
		continue;
	}

	return py::make_tuple(inputs, outputs, labels, num_collisions);
}

PYBIND11_MODULE(generator, m) {
	m.def("generate_training_set", &generate_training_set);
	m.def("generate_reachable_training_set", &generate_reachable_training_set);
	m.def("generate_dfs_training_set", &generate_dfs_training_set);
	m.def("set_seed", &core::set_seed);
	m.def("map_tokens_to_natural_language_batched", &map_tokens_to_natural_language_batched);
}

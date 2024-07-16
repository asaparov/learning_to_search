#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <core/array.h>
#include <core/random.h>

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
		in_degrees[i] = ALPHA + vertices[i].parents.length;
		out_degrees[i] = ALPHA + vertices[i].children.length;
	}
	for (unsigned int i = index; i < num_vertices; i++) {
		/* sample the number of child and parent vertices */
		unsigned int num_children = randrange(1, max_num_parents);
		unsigned int num_parents = randrange(num_children == 0 ? 1 : 0, max_num_parents);
		num_children = std::min(num_children, i);
		num_parents = std::min(num_parents, i);

		/* sample the children of this new node */
		array<float> probabilities(index);
		float total_probability = 0.0f;
		for (unsigned int j = 0; j < index; j++) {
			probabilities[j] = in_degrees[j];
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
			probabilities[j] = out_degrees[j];
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

py::tuple generate_training_set(const unsigned int max_input_size, const uint64_t dataset_size, const unsigned int max_lookahead, const unsigned int max_edges, const py::object& reserved_inputs, const int distance_from_start, const bool quiet=false)
{
	const unsigned int QUERY_PREFIX_TOKEN = (max_input_size-5) / 3 + 4;
	const unsigned int PADDING_TOKEN = (max_input_size-5) / 3 + 3;
	const unsigned int EDGE_PREFIX_TOKEN = (max_input_size-5) / 3 + 2;
	const unsigned int PATH_PREFIX_TOKEN = (max_input_size-5) / 3 + 1;

	unsigned int num_generated = 0;
	unsigned int num_collisions = 0;
	size_t input_shape[2]{dataset_size, max_input_size};
	size_t output_shape[1]{dataset_size};
	py::array_t<int64_t, py::array::c_style> inputs(input_shape);
	py::array_t<int64_t, py::array::c_style> outputs(output_shape);
	auto inputs_mem = inputs.mutable_unchecked<2>();
	auto outputs_mem = outputs.mutable_unchecked<1>();
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
	py::list valid_outputs;

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
				outputs_mem(num_generated) = choice(useful_steps.data, useful_steps.length)->id;
				py::list valid_output;
				for (node* n : useful_steps)
					valid_output.append(n->id);
				valid_outputs.append(valid_output);
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

	return py::make_tuple(inputs, outputs, valid_outputs, num_collisions);
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

PYBIND11_MODULE(generator, m) {
	m.def("generate_training_set", &generate_training_set);
	m.def("generate_reachable_training_set", &generate_reachable_training_set);
	m.def("set_seed", &core::set_seed);
}

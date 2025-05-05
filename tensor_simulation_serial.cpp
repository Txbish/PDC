#include <iostream>
#include <vector>
#include <complex>
#include <map>
#include <set>
#include <string>
#include <algorithm>
#include <numeric>   // For iota
#include <stdexcept> // For exceptions
#include <iomanip>   // For printing complex numbers
#include <chrono>    // For timings
#include <cmath>     // For sqrt
#include <queue>     // For BFS
#include <stack>     // For Girvan-Newman edge betweenness calculation

// --- Timer Utility ---
class Timer
{
public:
    Timer(const std::string &name) : name_(name), start_time_(std::chrono::high_resolution_clock::now()) {}
    ~Timer()
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_).count();
        std::cout << "[TIMER] " << name_ << ": " << duration << " ms\n";
    }

private:
    std::string name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

// --- Global Index Manager (Simple) ---
int next_index_id_global_counter = 0;
int generate_new_index_id()
{
    return next_index_id_global_counter++;
}

int next_tensor_id_global_counter = 0;
int generate_new_tensor_id()
{
    return next_tensor_id_global_counter++;
}

// --- Tensor Structure (from previous version, largely unchanged) ---
struct Tensor
{
    int id;
    std::string name;
    std::vector<std::complex<double>> data;
    std::vector<int> shape;
    std::vector<int> indices;

    Tensor() : id(generate_new_tensor_id()) {}

    size_t rank() const { return indices.size(); }
    size_t size() const
    {
        if (shape.empty())
            return 0; // Scalar represented by rank 0 but no shape
        size_t s = 1;
        for (int dim : shape)
            s *= dim;
        return s;
    }

    void print_info(bool print_data = false) const
    {
        std::cout << "Tensor ID: " << id << " (Name: " << name << "), Rank: " << rank() << "\n";
        std::cout << "  Shape: [";
        for (size_t i = 0; i < shape.size(); ++i)
            std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
        std::cout << "]\n";
        std::cout << "  Indices: [";
        for (size_t i = 0; i < indices.size(); ++i)
            std::cout << indices[i] << (i == indices.size() - 1 ? "" : ", ");
        std::cout << "]\n";
        if (print_data && data.size() < 17)
        { // Print data for small tensors
            std::cout << "  Data: [";
            for (size_t i = 0; i < data.size(); ++i)
            {
                std::cout << std::fixed << std::setprecision(2) << data[i]
                          << (i == data.size() - 1 ? "" : ", ");
            }
            std::cout << "]\n";
        }
    }
};

// --- Tensor Network Structure (from previous, with minor adjustments if needed) ---
struct TensorNetwork
{
    std::map<int, Tensor> tensors;
    std::set<int> open_indices;
    std::map<int, std::vector<std::pair<int, int>>> index_to_tensor_legs; // index_id -> list of (tensor_id, leg_pos_on_tensor)

    void add_tensor(const Tensor &t)
    {
        tensors[t.id] = t;
        for (size_t i = 0; i < t.indices.size(); ++i)
        {
            index_to_tensor_legs[t.indices[i]].push_back({t.id, static_cast<int>(i)});
        }
    }

    void remove_tensor(int tensor_id)
    {
        if (tensors.count(tensor_id))
        {
            const auto &tensor_to_remove = tensors.at(tensor_id);
            for (size_t i = 0; i < tensor_to_remove.indices.size(); ++i)
            {
                int index_id = tensor_to_remove.indices[i];
                if (index_to_tensor_legs.count(index_id))
                {
                    auto &leg_list = index_to_tensor_legs.at(index_id);
                    leg_list.erase(std::remove_if(leg_list.begin(), leg_list.end(),
                                                  [&](const std::pair<int, int> &p)
                                                  { return p.first == tensor_id; }),
                                   leg_list.end());
                    if (leg_list.empty())
                    {
                        index_to_tensor_legs.erase(index_id);
                        open_indices.erase(index_id);
                    }
                }
            }
            tensors.erase(tensor_id);
        }
    }

    // get_shared_unique_indices (from previous)
    void get_shared_unique_indices(
        const Tensor &t1, const Tensor &t2,
        std::vector<int> &shared_indices_val,            // The actual index IDs
        std::vector<std::pair<int, int>> &t1_shared_map, // {t1_leg_idx, pos_in_shared_indices_val}
        std::vector<std::pair<int, int>> &t2_shared_map, // {t2_leg_idx, pos_in_shared_indices_val}
        std::vector<int> &t1_unique_indices, std::vector<int> &t1_unique_dims,
        std::vector<int> &t2_unique_indices, std::vector<int> &t2_unique_dims) const
    {
        shared_indices_val.clear();
        t1_shared_map.clear();
        t2_shared_map.clear();
        t1_unique_indices.clear();
        t1_unique_dims.clear();
        t2_unique_indices.clear();
        t2_unique_dims.clear();

        std::set<int> s1_indices(t1.indices.begin(), t1.indices.end());
        std::set<int> s2_indices(t2.indices.begin(), t2.indices.end());

        std::set_intersection(s1_indices.begin(), s1_indices.end(),
                              s2_indices.begin(), s2_indices.end(),
                              std::back_inserter(shared_indices_val));

        for (size_t i = 0; i < t1.indices.size(); ++i)
        {
            auto it = std::find(shared_indices_val.begin(), shared_indices_val.end(), t1.indices[i]);
            if (it != shared_indices_val.end())
            {
                t1_shared_map.push_back({(int)i, (int)std::distance(shared_indices_val.begin(), it)});
            }
            else
            {
                t1_unique_indices.push_back(t1.indices[i]);
                t1_unique_dims.push_back(t1.shape[i]);
            }
        }

        for (size_t i = 0; i < t2.indices.size(); ++i)
        {
            auto it = std::find(shared_indices_val.begin(), shared_indices_val.end(), t2.indices[i]);
            if (it != shared_indices_val.end())
            {
                t2_shared_map.push_back({(int)i, (int)std::distance(shared_indices_val.begin(), it)});
            }
            else
            {
                t2_unique_indices.push_back(t2.indices[i]);
                t2_unique_dims.push_back(t2.shape[i]);
            }
        }
    }

    // contract_tensors (from previous, critical and complex)
    Tensor contract_tensors(const Tensor &t1, const Tensor &t2, const std::string &new_name = "Contracted") const
    {
        std::vector<int> shared_indices_list;
        std::vector<std::pair<int, int>> t1_shared_map, t2_shared_map;
        std::vector<int> t1_unique_indices_list, t2_unique_indices_list;
        std::vector<int> t1_unique_dims, t2_unique_dims;

        get_shared_unique_indices(t1, t2, shared_indices_list, t1_shared_map, t2_shared_map,
                                  t1_unique_indices_list, t1_unique_dims,
                                  t2_unique_indices_list, t2_unique_dims);
        Tensor result;
        result.name = new_name;
        result.indices.insert(result.indices.end(), t1_unique_indices_list.begin(), t1_unique_indices_list.end());
        result.indices.insert(result.indices.end(), t2_unique_indices_list.begin(), t2_unique_indices_list.end());
        result.shape.insert(result.shape.end(), t1_unique_dims.begin(), t1_unique_dims.end());
        result.shape.insert(result.shape.end(), t2_unique_dims.begin(), t2_unique_dims.end());

        size_t result_size = 1;
        if (result.shape.empty() && !result.indices.empty())
        {                    // e.g. trace of a matrix gives scalar with 0 indices
            result_size = 1; // Scalar
        }
        else
        {
            for (int dim : result.shape)
                result_size *= dim;
        }
        if (result_size == 0 && result.rank() > 0)
            throw std::runtime_error("Result tensor has zero size but non-zero rank");
        if (result_size == 0 && result.rank() == 0)
            result_size = 1; // scalar from full contraction

        result.data.resize(result_size, {0.0, 0.0});

        std::vector<size_t> t1_strides(t1.rank()), t2_strides(t2.rank()), res_strides(result.rank());
        if (t1.rank() > 0)
        {
            t1_strides.back() = 1;
            for (int k = t1.rank() - 2; k >= 0; --k)
                t1_strides[k] = t1_strides[k + 1] * t1.shape[k + 1];
        }
        if (t2.rank() > 0)
        {
            t2_strides.back() = 1;
            for (int k = t2.rank() - 2; k >= 0; --k)
                t2_strides[k] = t2_strides[k + 1] * t2.shape[k + 1];
        }
        if (result.rank() > 0)
        {
            res_strides.back() = 1;
            for (int k = result.rank() - 2; k >= 0; --k)
                res_strides[k] = res_strides[k + 1] * result.shape[k + 1];
        }

        auto get_flat_idx = [&](const std::vector<int> &multi_idx, const std::vector<size_t> &strides, const std::vector<int> & /*shape_vec*/)
        {
            size_t flat_idx = 0;
            for (size_t k = 0; k < multi_idx.size(); ++k)
            {
                flat_idx += multi_idx[k] * strides[k];
            }
            return flat_idx;
        };

        std::vector<int> current_t1_multi_idx(t1.rank(), 0);
        std::vector<int> current_res_multi_idx(result.rank(), 0);

        for (size_t i_flat_t1 = 0; i_flat_t1 < t1.size(); ++i_flat_t1)
        {
            size_t temp_i = i_flat_t1;
            for (int k_t1 = t1.rank() - 1; k_t1 >= 0; --k_t1)
            {
                current_t1_multi_idx[k_t1] = temp_i % t1.shape[k_t1];
                temp_i /= t1.shape[k_t1];
            }
            // if (t1.rank() > 0) std::reverse(current_t1_multi_idx.begin(), current_t1_multi_idx.end()); // No, strides handle order

            std::vector<int> current_t2_multi_idx(t2.rank(), 0);
            for (size_t j_flat_t2 = 0; j_flat_t2 < t2.size(); ++j_flat_t2)
            {
                size_t temp_j = j_flat_t2;
                for (int k_t2 = t2.rank() - 1; k_t2 >= 0; --k_t2)
                {
                    current_t2_multi_idx[k_t2] = temp_j % t2.shape[k_t2];
                    temp_j /= t2.shape[k_t2];
                }
                // if (t2.rank() > 0) std::reverse(current_t2_multi_idx.begin(), current_t2_multi_idx.end());

                bool match = true;
                for (const auto &t1_map_entry : t1_shared_map)
                { // {t1_leg, shared_idx_pos}
                    int t1_leg = t1_map_entry.first;
                    int shared_idx_pos = t1_map_entry.second;
                    int t2_leg_for_this_shared = -1;
                    for (const auto &t2_map_entry : t2_shared_map)
                    {
                        if (t2_map_entry.second == shared_idx_pos)
                        {
                            t2_leg_for_this_shared = t2_map_entry.first;
                            break;
                        }
                    }
                    if (t2_leg_for_this_shared == -1)
                        throw std::runtime_error("Shared index consistency error");
                    if (current_t1_multi_idx[t1_leg] != current_t2_multi_idx[t2_leg_for_this_shared])
                    {
                        match = false;
                        break;
                    }
                }

                if (match)
                {
                    size_t res_k = 0;
                    for (size_t k_t1_unique = 0; k_t1_unique < t1_unique_indices_list.size(); ++k_t1_unique)
                    {
                        int t1_orig_leg = -1;
                        for (size_t leg = 0; leg < t1.indices.size(); ++leg)
                        {
                            if (t1.indices[leg] == t1_unique_indices_list[k_t1_unique])
                            {
                                t1_orig_leg = leg;
                                break;
                            }
                        }
                        if (t1_orig_leg == -1)
                            throw std::runtime_error("Unique T1 mapping error");
                        current_res_multi_idx[res_k++] = current_t1_multi_idx[t1_orig_leg];
                    }
                    for (size_t k_t2_unique = 0; k_t2_unique < t2_unique_indices_list.size(); ++k_t2_unique)
                    {
                        int t2_orig_leg = -1;
                        for (size_t leg = 0; leg < t2.indices.size(); ++leg)
                        {
                            if (t2.indices[leg] == t2_unique_indices_list[k_t2_unique])
                            {
                                t2_orig_leg = leg;
                                break;
                            }
                        }
                        if (t2_orig_leg == -1)
                            throw std::runtime_error("Unique T2 mapping error");
                        current_res_multi_idx[res_k++] = current_t2_multi_idx[t2_orig_leg];
                    }

                    size_t flat_res_idx = 0;
                    if (result.rank() > 0)
                        flat_res_idx = get_flat_idx(current_res_multi_idx, res_strides, result.shape);
                    else if (result.rank() == 0 && result_size == 1)
                        flat_res_idx = 0; // Scalar case
                    else if (result.rank() == 0 && result_size == 0)
                        continue; // Should not happen for valid contraction

                    if (flat_res_idx >= result.data.size())
                    {
                        throw std::out_of_range("Result index out of bounds. Res Idx: " + std::to_string(flat_res_idx) + ", Res Data Size: " + std::to_string(result.data.size()));
                    }
                    result.data[flat_res_idx] += t1.data[i_flat_t1] * t2.data[j_flat_t2];
                }
            }
        }
        return result;
    }

    void contract_pair_in_network(int t1_id, int t2_id)
    {
        if (!tensors.count(t1_id) || !tensors.count(t2_id))
        {
            std::cerr << "Error: Tensors for contraction not found: " << t1_id << ", " << t2_id << std::endl;
            throw std::runtime_error("Tensor not found for contraction");
        }
        Tensor contracted_tensor = contract_tensors(tensors.at(t1_id), tensors.at(t2_id));

        std::set<int> t1_indices_set(tensors.at(t1_id).indices.begin(), tensors.at(t1_id).indices.end());
        std::set<int> t2_indices_set(tensors.at(t2_id).indices.begin(), tensors.at(t2_id).indices.end());
        std::vector<int> shared;
        std::set_intersection(t1_indices_set.begin(), t1_indices_set.end(),
                              t2_indices_set.begin(), t2_indices_set.end(),
                              std::back_inserter(shared));
        for (int idx : shared)
        {
            open_indices.erase(idx); // These are contracted away
            // Also remove from index_to_tensor_legs as they are no longer "connecting" different tensors
            index_to_tensor_legs.erase(idx);
        }

        remove_tensor(t1_id); // Must remove before adding, to avoid issues with index_to_tensor_legs
        remove_tensor(t2_id);
        add_tensor(contracted_tensor);

        // Re-evaluate open_indices based on the new tensor and remaining network
        // This is complex. For now, the add_tensor and remove_tensor should handle index_to_tensor_legs.
        // Open indices are those connected to only one tensor leg in the *entire* network.
        // The current logic for open_indices in remove_tensor and after contraction needs to be robust.
        // Let's recalculate open_indices for the whole network after a contraction
        std::set<int> new_open_indices;
        for (const auto &pair_idx_legs : index_to_tensor_legs)
        {
            if (pair_idx_legs.second.size() == 1)
            { // Only one tensor leg attached to this index
                new_open_indices.insert(pair_idx_legs.first);
            }
        }
        open_indices = new_open_indices;
    }

    void print_summary(bool print_tensor_data = false) const
    {
        std::cout << "\n--- Tensor Network Summary ---\n";
        std::cout << "Number of tensors: " << tensors.size() << "\n";
        for (const auto &pair : tensors)
        {
            pair.second.print_info(print_tensor_data);
        }
        std::cout << "Open Indices (" << open_indices.size() << "): [";
        for (int idx : open_indices)
            std::cout << idx << " ";
        std::cout << "]\n";
        // std::cout << "Index to Tensor Legs map:\n";
        // for(const auto& pair : index_to_tensor_legs) {
        //     std::cout << "  Index " << pair.first << ": ";
        //     for(const auto& leg : pair.second) std::cout << "(T" << leg.first << ",L" << leg.second << ") ";
        //     std::cout << "\n";
        // }
        std::cout << "-----------------------------\n";
    }
};

// --- Quantum Gate Factories (from previous) ---
Tensor create_H_gate()
{
    Tensor h;
    h.name = "H";
    h.shape = {2, 2};
    h.indices = {generate_new_index_id(), generate_new_index_id()};
    h.data = {{1 / sqrt(2.0), 0}, {1 / sqrt(2.0), 0}, {1 / sqrt(2.0), 0}, {-1 / sqrt(2.0), 0}};
    return h;
}
Tensor create_CNOT_gate()
{
    Tensor cnot;
    cnot.name = "CNOT";
    cnot.shape = {2, 2, 2, 2};
    cnot.indices = {generate_new_index_id(), generate_new_index_id(), generate_new_index_id(), generate_new_index_id()};
    cnot.data.assign(16, {0, 0});
    cnot.data[0b0000] = {1, 0};
    cnot.data[0b0101] = {1, 0};
    cnot.data[0b1110] = {1, 0};
    cnot.data[0b1011] = {1, 0}; // out_c, out_t, in_c, in_t
    return cnot;
}
Tensor create_qubit_state_zero()
{
    Tensor q0;
    q0.name = "|0>";
    q0.shape = {2};
    q0.indices = {generate_new_index_id()};
    q0.data = {{1, 0}, {0, 0}};
    return q0;
}

// --- GHZ Circuit to Tensor Network ---
TensorNetwork create_ghz_circuit_tn(int num_qubits)
{
    if (num_qubits < 1)
        throw std::invalid_argument("Number of qubits must be at least 1 for GHZ.");
    Timer timer("GHZ TN Creation (" + std::to_string(num_qubits) + " qubits)");
    TensorNetwork tn;
    next_index_id_global_counter = 0;
    next_tensor_id_global_counter = 0;

    std::vector<int> qubit_lines(num_qubits);
    for (int i = 0; i < num_qubits; ++i)
    {
        Tensor s = create_qubit_state_zero();
        s.name = "|0>_q" + std::to_string(i);
        // The initial index of this state tensor becomes the first segment of the qubit line
        qubit_lines[i] = s.indices[0];
        tn.add_tensor(s);
    }

    // H on q0
    if (num_qubits > 0)
    {
        Tensor h_q0 = create_H_gate();
        h_q0.name = "H_q0";
        int q0_line_after_h = generate_new_index_id();
        h_q0.indices = {q0_line_after_h, qubit_lines[0]}; // out, in
        tn.add_tensor(h_q0);
        qubit_lines[0] = q0_line_after_h; // Update current end of qubit line 0
    }

    // Chain of CNOTs: CNOT(q_i, q_{i+1})
    for (int i = 0; i < num_qubits - 1; ++i)
    {
        Tensor cnot_gate = create_CNOT_gate();
        cnot_gate.name = "CNOT_q" + std::to_string(i) + "_q" + std::to_string(i + 1);

        int ctrl_out_idx = generate_new_index_id();
        int tgt_out_idx = generate_new_index_id();

        // Indices: out_ctrl, out_tgt, in_ctrl, in_tgt
        cnot_gate.indices = {ctrl_out_idx, tgt_out_idx, qubit_lines[i], qubit_lines[i + 1]};
        tn.add_tensor(cnot_gate);

        qubit_lines[i] = ctrl_out_idx;    // Update current end of control qubit line
        qubit_lines[i + 1] = tgt_out_idx; // Update current end of target qubit line
    }

    // Set open indices (the final ends of all qubit lines)
    for (int line_idx : qubit_lines)
    {
        tn.open_indices.insert(line_idx);
    }

    return tn;
}

// --- Girvan-Newman Community Detection ---
namespace GirvanNewman
{
    using GraphAdj = std::map<int, std::set<int>>; // Tensor ID -> Set of connected Tensor IDs
    using Edge = std::pair<int, int>;              // Sorted (min_id, max_id) to uniquely represent edges

    struct EdgeHash
    {
        std::size_t operator()(const Edge &edge) const
        {
            return std::hash<int>()(edge.first) ^ (std::hash<int>()(edge.second) << 1);
        }
    };

    // BFS to calculate shortest paths, distances, and predecessors
    void bfs(int start_node, const GraphAdj &adj,
             std::map<int, int> &dist,
             std::map<int, double> &num_shortest_paths,
             std::map<int, std::vector<int>> &predecessors)
    {
        dist.clear();
        num_shortest_paths.clear();
        predecessors.clear();

        for (const auto &pair : adj)
        { // Initialize for all nodes in graph
            dist[pair.first] = -1;
            num_shortest_paths[pair.first] = 0;
        }
        if (!adj.count(start_node))
            return;

        std::queue<int> q;
        q.push(start_node);
        dist[start_node] = 0;
        num_shortest_paths[start_node] = 1;

        std::stack<int> visited_order; // For betweenness calculation order

        while (!q.empty())
        {
            int u = q.front();
            q.pop();
            visited_order.push(u);

            if (adj.count(u))
            {
                for (int v : adj.at(u))
                {
                    if (dist[v] == -1)
                    { // First time reaching v
                        dist[v] = dist[u] + 1;
                        q.push(v);
                    }
                    if (dist[v] == dist[u] + 1)
                    { // Shortest path to v via u
                        num_shortest_paths[v] += num_shortest_paths[u];
                        predecessors[v].push_back(u);
                    }
                }
            }
        }
    }

    // Calculate edge betweenness
    std::map<Edge, double> calculate_edge_betweenness(const std::set<int> &nodes, const GraphAdj &adj)
    {
        std::map<Edge, double> edge_betweenness;
        for (const auto &pair_u : adj)
        {
            for (int v : pair_u.second)
            {
                if (pair_u.first < v)
                    edge_betweenness[{pair_u.first, v}] = 0.0;
            }
        }

        for (int s : nodes)
        {
            std::map<int, int> dist;
            std::map<int, double> num_shortest_paths;
            std::map<int, std::vector<int>> predecessors_map; // Renamed to avoid conflict

            std::stack<int> S; // Stack of nodes in order of BFS discovery (farthest first for accumulation)
            std::queue<int> Q_bfs;

            for (int node_id : nodes)
            {
                dist[node_id] = -1;
                num_shortest_paths[node_id] = 0;
            }

            dist[s] = 0;
            num_shortest_paths[s] = 1.0;
            Q_bfs.push(s);

            while (!Q_bfs.empty())
            {
                int v = Q_bfs.front();
                Q_bfs.pop();
                S.push(v);
                if (adj.count(v))
                {
                    for (int w : adj.at(v))
                    {
                        if (dist[w] < 0)
                        {
                            Q_bfs.push(w);
                            dist[w] = dist[v] + 1;
                        }
                        if (dist[w] == dist[v] + 1)
                        {
                            num_shortest_paths[w] += num_shortest_paths[v];
                            predecessors_map[w].push_back(v);
                        }
                    }
                }
            }

            std::map<int, double> dependency;
            for (int node_id : nodes)
                dependency[node_id] = 0.0;

            while (!S.empty())
            {
                int w = S.top();
                S.pop();
                if (predecessors_map.count(w))
                {
                    for (int v : predecessors_map.at(w))
                    {
                        if (num_shortest_paths[w] == 0)
                            continue; // Avoid division by zero
                        double credit = (num_shortest_paths[v] / num_shortest_paths[w]) * (1.0 + dependency[w]);
                        Edge current_edge = {std::min(v, w), std::max(v, w)};
                        edge_betweenness[current_edge] += credit;
                        dependency[v] += credit;
                    }
                }
            }
        }
        return edge_betweenness;
    }

    // Get connected components
    std::vector<std::vector<int>> get_connected_components(const std::set<int> &nodes, const GraphAdj &adj)
    {
        std::vector<std::vector<int>> components;
        std::set<int> visited;
        for (int node : nodes)
        {
            if (visited.find(node) == visited.end())
            {
                std::vector<int> current_component;
                std::queue<int> q;
                q.push(node);
                visited.insert(node);
                current_component.push_back(node);
                while (!q.empty())
                {
                    int u = q.front();
                    q.pop();
                    if (adj.count(u))
                    {
                        for (int v : adj.at(u))
                        {
                            if (visited.find(v) == visited.end())
                            {
                                visited.insert(v);
                                q.push(v);
                                current_component.push_back(v);
                            }
                        }
                    }
                }
                components.push_back(current_component);
            }
        }
        return components;
    }

    // Main Girvan-Newman function
    std::vector<std::vector<int>> detect_communities(const TensorNetwork &tn, int num_target_communities)
    {
        Timer timer_gn("Girvan-Newman Community Detection");
        // Build graph: nodes are tensor IDs, edges if they share an index
        std::set<int> graph_nodes;
        GraphAdj graph_adj;
        std::set<Edge> graph_edges;

        for (const auto &pair_t : tn.tensors)
        {
            graph_nodes.insert(pair_t.first);
            graph_adj[pair_t.first] = {}; // Initialize adjacency list
        }

        for (const auto &pair_idx_legs : tn.index_to_tensor_legs)
        {
            const auto &legs = pair_idx_legs.second;
            for (size_t i = 0; i < legs.size(); ++i)
            {
                for (size_t j = i + 1; j < legs.size(); ++j)
                {
                    int t1_id = legs[i].first;
                    int t2_id = legs[j].first;
                    if (t1_id != t2_id)
                    { // An index connects two different tensors
                        graph_adj[t1_id].insert(t2_id);
                        graph_adj[t2_id].insert(t1_id);
                        graph_edges.insert({std::min(t1_id, t2_id), std::max(t1_id, t2_id)});
                    }
                }
            }
        }

        if (graph_nodes.empty())
            return {};
        if (num_target_communities <= 0)
            num_target_communities = 1;
        if (num_target_communities > (int)graph_nodes.size())
            num_target_communities = graph_nodes.size();

        GraphAdj current_adj = graph_adj;
        std::set<int> current_nodes = graph_nodes;
        std::vector<std::vector<int>> components = get_connected_components(current_nodes, current_adj);
        int iteration = 0;

        while (components.size() < (size_t)num_target_communities && !graph_edges.empty())
        {
            iteration++;
            // std::cout << "GN Iteration: " << iteration << ", Num Components: " << components.size() << ", Num Edges: " << graph_edges.size() << std::endl;
            std::map<Edge, double> edge_betweenness = calculate_edge_betweenness(current_nodes, current_adj);
            if (edge_betweenness.empty())
                break; // No more edges to remove or graph disconnected unexpectedly

            Edge edge_to_remove = {-1, -1};
            double max_betweenness = -1.0;
            for (const auto &pair_edge_bw : edge_betweenness)
            {
                if (pair_edge_bw.second > max_betweenness)
                {
                    max_betweenness = pair_edge_bw.second;
                    edge_to_remove = pair_edge_bw.first;
                }
            }

            if (edge_to_remove.first == -1)
                break; // No edge found (e.g., fully disconnected)

            // Remove edge
            current_adj[edge_to_remove.first].erase(edge_to_remove.second);
            current_adj[edge_to_remove.second].erase(edge_to_remove.first);
            graph_edges.erase(edge_to_remove); // Keep track of overall edges too

            components = get_connected_components(current_nodes, current_adj);
            if (components.empty() && !current_nodes.empty())
            { // Should not happen if nodes exist
                std::cerr << "Warning: GN resulted in empty components with non-empty nodes.\n";
                break;
            }
        }
        return components;
    }
} // namespace GirvanNewman

// --- Subnetwork Extraction and Contraction (from previous, may need minor adjustments) ---
TensorNetwork extract_subnetwork(const TensorNetwork &original_tn, const std::vector<int> &tensor_ids_in_community)
{
    TensorNetwork sub_tn;
    std::set<int> all_indices_in_community_tensors;

    for (int id : tensor_ids_in_community)
    {
        if (original_tn.tensors.count(id))
        {
            sub_tn.add_tensor(original_tn.tensors.at(id));
            for (int idx_id : original_tn.tensors.at(id).indices)
            {
                all_indices_in_community_tensors.insert(idx_id);
            }
        }
    }

    // Determine open indices for the sub-network
    for (int idx_id : all_indices_in_community_tensors)
    {
        if (!original_tn.index_to_tensor_legs.count(idx_id))
            continue; // Should not happen if index came from a tensor

        const auto &connections = original_tn.index_to_tensor_legs.at(idx_id);
        bool is_external_to_community = false;
        if (connections.empty())
            continue; // Should not happen

        for (const auto &leg_pair : connections)
        { // (tensor_id, leg_pos)
            bool leg_tensor_in_this_community = false;
            for (int comm_tensor_id : tensor_ids_in_community)
            {
                if (leg_pair.first == comm_tensor_id)
                {
                    leg_tensor_in_this_community = true;
                    break;
                }
            }
            if (!leg_tensor_in_this_community)
            { // This index connects to a tensor *outside* this community
                is_external_to_community = true;
                break;
            }
        }
        // An index is also open if it's an original open index of the main TN *and* belongs to one of the community's tensors
        bool is_original_open_and_in_community = original_tn.open_indices.count(idx_id);

        if (is_external_to_community || is_original_open_and_in_community)
        {
            sub_tn.open_indices.insert(idx_id);
        }
    }
    return sub_tn;
}

Tensor contract_subnetwork_sequentially(TensorNetwork &sub_tn, const std::string &name_prefix)
{
    int contract_count = 0;
    while (sub_tn.tensors.size() > 1)
    {
        if (sub_tn.tensors.empty())
            throw std::runtime_error("Cannot contract empty subnetwork");

        // Naive: find any two tensors that share an index
        int id1 = -1, id2 = -1;
        for (const auto &pair_t1 : sub_tn.tensors)
        {
            for (const auto &pair_t2 : sub_tn.tensors)
            {
                if (pair_t1.first == pair_t2.first)
                    continue;
                std::vector<int> shared_indices_val;
                std::vector<std::pair<int, int>> t1_shared_map, t2_shared_map;
                std::vector<int> t1_unique_indices, t1_unique_dims;
                std::vector<int> t2_unique_indices, t2_unique_dims;
                sub_tn.get_shared_unique_indices(pair_t1.second, pair_t2.second, shared_indices_val,
                                                 t1_shared_map, t2_shared_map, t1_unique_indices, t1_unique_dims,
                                                 t2_unique_indices, t2_unique_dims);
                if (!shared_indices_val.empty())
                {
                    id1 = pair_t1.first;
                    id2 = pair_t2.first;
                    goto found_pair;
                }
            }
        }
    found_pair:;

        if (id1 == -1)
        { // No pair shares indices, could be multiple disconnected tensors
            // This indicates an issue with subnetwork formation or contraction strategy.
            // For now, just take the first one if multiple disconnected remain.
            std::cout << "Warning: Subnetwork has " << sub_tn.tensors.size() << " tensors but no shared indices found for contraction. "
                      << "This might mean the subnetwork is already disconnected components." << std::endl;
            sub_tn.print_summary(true);
            // Just break and return the "largest" or first tensor if this happens
            // This is a fallback, ideal contraction plan would avoid this.
            if (!sub_tn.tensors.empty())
                return sub_tn.tensors.begin()->second;
            else
                throw std::runtime_error("Subnetwork contraction failed: no tensors left after finding no shared indices.");
        }

        // std::cout << "  Contracting in subnetwork: " << name_prefix << ", Tensors " << id1 << " and " << id2 << std::endl;
        std::string contracted_name = name_prefix + "_c" + std::to_string(contract_count++);
        Tensor contracted = sub_tn.contract_tensors(sub_tn.tensors.at(id1), sub_tn.tensors.at(id2), contracted_name);

        sub_tn.remove_tensor(id1);
        sub_tn.remove_tensor(id2);
        sub_tn.add_tensor(contracted);
    }
    if (sub_tn.tensors.size() != 1)
    {
        std::cout << "Warning: Subnetwork contraction did not result in a single tensor for " << name_prefix
                  << ". Remaining tensors: " << sub_tn.tensors.size() << std::endl;
        if (sub_tn.tensors.empty())
            throw std::runtime_error("Subnetwork contraction resulted in zero tensors for " + name_prefix);
        return sub_tn.tensors.begin()->second; // Return first if multiple disconnected parts remain
    }
    return sub_tn.tensors.begin()->second;
}

// --- Main ComPar Algorithm (Serial Version) ---
int main(int argc, char *argv[])
{
    int num_qubits = 3;             // Default
    int num_communities_target = 2; // Default

    if (argc > 1)
        num_qubits = std::stoi(argv[1]);
    if (argc > 2)
        num_communities_target = std::stoi(argv[2]);

    std::cout << "--- Quantum Circuit Simulation using Tensor Networks (Serial ComPar) ---\n";
    std::cout << "Config: " << num_qubits << " qubits, aiming for " << num_communities_target << " communities.\n";
    Timer total_sim_timer("Total Simulation");

    // 1. Create Initial Tensor Network (GHZ circuit)
    TensorNetwork original_tn;
    {
        Timer timer_tn_creation("Tensor Network Creation");
        original_tn = create_ghz_circuit_tn(num_qubits);
    }
    // original_tn.print_summary(true); // Print with data for small circuits

    // STAGE 1: Partitioning into communities using Girvan-Newman
    std::vector<std::vector<int>> communities;
    {
        Timer timer_stage1("Stage 1: Community Detection (Girvan-Newman)");
        communities = GirvanNewman::detect_communities(original_tn, num_communities_target);
    }

    std::cout << "[STAGE 1] Detected " << communities.size() << " communities:\n";
    for (size_t i = 0; i < communities.size(); ++i)
    {
        std::cout << "  Community " << i << " (Tensor IDs): [";
        for (size_t j = 0; j < communities[i].size(); ++j)
        {
            std::cout << communities[i][j] << (j == communities[i].size() - 1 ? "" : ", ");
        }
        std::cout << "]\n";
    }

    // STAGE 2: Contracting sub-tensor networks (serially)
    TensorNetwork network_of_communities_tn;
    {
        Timer timer_stage2("Stage 2: Sub-network Contractions");
        for (size_t i = 0; i < communities.size(); ++i)
        {
            // std::cout << "  Processing Community " << i << "...\n";
            std::vector<int> tensor_ids_in_community = communities[i];

            if (tensor_ids_in_community.empty())
            {
                // std::cout << "  Community " << i << " is empty, skipping.\n";
                continue;
            }

            TensorNetwork sub_tn = extract_subnetwork(original_tn, tensor_ids_in_community);
            // std::cout << "  Sub-network for community " << i << ":\n";
            // sub_tn.print_summary(num_qubits <=3); // Print data for small subnets

            if (sub_tn.tensors.empty())
            {
                // std::cout << "  Sub-network for community " << i << " is empty after extraction, skipping.\n";
                continue;
            }

            Tensor community_result;
            if (sub_tn.tensors.size() == 1)
            {
                // std::cout << "  Community " << i << " has only one tensor. Adding it directly.\n";
                community_result = sub_tn.tensors.begin()->second;
            }
            else
            {
                community_result = contract_subnetwork_sequentially(sub_tn, "Comm" + std::to_string(i));
            }
            community_result.name = "CommunityRes_" + std::to_string(i);
            // std::cout << "  Community " << i << " contracted to:\n";
            // community_result.print_info(num_qubits <=3);
            network_of_communities_tn.add_tensor(community_result);
        }
    }

    // std::cout << "\nNetwork of communities built from contracted sub-networks:\n";
    // network_of_communities_tn.print_summary(num_qubits <=3);

    // STAGE 3: Contracting the resulting network of communities
    Tensor final_result_tensor;
    bool final_contraction_done = false;
    {
        Timer timer_stage3("Stage 3: Final Community Network Contraction");
        if (network_of_communities_tn.tensors.empty())
        {
            std::cout << "No community tensors to contract. Final result is effectively empty.\n";
        }
        else if (network_of_communities_tn.tensors.size() == 1)
        {
            // std::cout << "Network of communities has only one tensor. This is the final result.\n";
            final_result_tensor = network_of_communities_tn.tensors.begin()->second;
            final_contraction_done = true;
        }
        else
        {
            final_result_tensor = contract_subnetwork_sequentially(network_of_communities_tn, "Final");
            final_contraction_done = true;
        }
    }

    if (final_contraction_done)
    {
        std::cout << "\n--- Final Result Tensor --- (for " << num_qubits << " qubits GHZ)\n";
        final_result_tensor.print_info(true); // Print data for the final scalar/vector
    }
    else
    {
        std::cout << "\n--- Final contraction skipped or failed ---\n";
    }

    std::cout << "\n--- Simulation Complete ---\n";
    return 0;
}
#include "incremental_nns.h"
#include "time_measure_util.h"
#include <limits>
#include <cassert>
#include <cstddef>
#include <algorithm>

namespace DENSE_MULTICUT {

    incremental_nns::incremental_nns(
        const std::vector<faiss::Index::idx_t>& query_nodes, const std::vector<faiss::Index::idx_t>& nns, const std::vector<float>& nns_distances, const size_t n, const size_t k)
    {
        // Store as undirected graph.
        nn_graph_ = std::vector<std::unordered_map<size_t, float>>(2 * n);
        min_dist_in_knn_ = std::vector<float>(2 * n, std::numeric_limits<float>::infinity());
        k_ = k;
        insert_nn_to_graph(query_nodes, nns, nns_distances, k);
    }

    void incremental_nns::insert_nn_to_graph(
        const std::vector<faiss::Index::idx_t>& query_nodes, const std::vector<faiss::Index::idx_t>& nns, const std::vector<float>& nns_distances, const size_t k)
    {
        size_t index_1d = 0;
        for (size_t idx = 0; idx != query_nodes.size(); ++idx)
        {
            const size_t i = query_nodes[idx];
            for (size_t i_n = 0; i_n != k; ++i_n, ++index_1d)
            {
                const float current_distance = nns_distances[index_1d];
                if (current_distance < 0)
                    continue;

                const size_t j = nns[index_1d];
                nn_graph_[i].try_emplace(j, current_distance);
                nn_graph_[j].try_emplace(i, current_distance);
                min_dist_in_knn_[i] = std::min(min_dist_in_knn_[i], current_distance);
                min_dist_in_knn_[j] = std::min(min_dist_in_knn_[j], current_distance);
            }
        }
    }

    std::unordered_map<size_t, float> incremental_nns::merge_nodes(const size_t i, const size_t j, const size_t new_id, const feature_index& index, const bool do_exhaustive_search)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
        const size_t root = nn_graph_[i].size() >= nn_graph_[j].size() ? i: j;
        const size_t other = root == i ? j : i;
        
        const size_t current_k = 10 * k_; // * index.nr_nodes_in_cluster(i) * index.nr_nodes_in_cluster(j);
        std::vector<std::pair<size_t, float>> nn_ij;

        const float upper_bound_outside_knn_ij = min_dist_in_knn_[i] + min_dist_in_knn_[j];

        float largest_distance = 0.0;
        // iterate over kNNs of root:
        for (auto const& [nn_root, cost_root] : nn_graph_[root])
        {
            if (nn_root == other)
                continue;
            // check if nn_root is also in kNN(other):
            const auto nn_other_iter = nn_graph_[other].find(nn_root);
            if (nn_other_iter != nn_graph_[other].end() && nn_ij.size() < current_k)
            {
                const float current_dist = cost_root + nn_other_iter->second;
                assert(current_dist >= upper_bound_outside_knn_ij);
                largest_distance = std::max(largest_distance, current_dist);
                nn_ij.push_back({nn_root, current_dist});
            }
        }

        for (auto const& [nn_root, cost_root] : nn_graph_[root])
        {
            if (nn_root == other)
                continue;
            const auto nn_other_iter = nn_graph_[other].find(nn_root);
            if (nn_other_iter == nn_graph_[other].end() && nn_ij.size() < current_k)
            {
                // Compute cost between other and nn_root and add.
                const float new_dist = cost_root + index.inner_product(nn_root, other); 
                if (new_dist >= upper_bound_outside_knn_ij)
                {
                    largest_distance = std::max(largest_distance, new_dist);
                    nn_ij.push_back({nn_root, new_dist});
                }
            }
            // Remove root as neighbour of nn_root:
            nn_graph_[nn_root].erase(root);
        }

        // Now iterate over kNNs of other:
        for (auto const& [nn_other, cost_other] : nn_graph_[other])
        {
            if (nn_other == root)
                continue;
            // Skip if nn_other is also in kNN(root) as already considered above.
            if (nn_graph_[root].find(nn_other) == nn_graph_[root].end() && nn_ij.size() < current_k)
            {
                const float new_dist = cost_other + index.inner_product(nn_other, root); // Compute cost between root and nn_other and add.
                if (new_dist>= upper_bound_outside_knn_ij)
                {
                    largest_distance = std::max(largest_distance, new_dist);
                    nn_ij.push_back({nn_other, new_dist});
                }
            }
            // Remove other as neighbour of nn_other:
            nn_graph_[nn_other].erase(other);
        }
            
        // If no new neighbours are found within KNNs of i and j, then search in whole graph for current_k many nearest neighbours.
        if (do_exhaustive_search && (nn_ij.size() == 0 || largest_distance < upper_bound_outside_knn_ij) && index.nr_nodes() > 1)
        {
            const std::vector<faiss::Index::idx_t> new_id_to_search = {new_id};
            const auto [nns, distances] = index.get_nearest_nodes(new_id_to_search, std::min(current_k, index.nr_nodes() - 1));
            for (int idx = 0; idx != nns.size(); ++idx)
            {
                const float current_distance = distances[idx];
                if (current_distance >= 0.0)
                    nn_ij.push_back({nns[idx], current_distance});
            }
            std::cout<<"[incremental nns] Performing exhaustive search on "<<index.nr_nodes()<<" nodes. ";
            std::cout<<"Found inc. neighbours: "<<nn_ij.size()<<", with max. cost: "<<largest_distance<<", UB: "<<upper_bound_outside_knn_ij<<"\n";
        }

        // Remove root and other nodes.
        nn_graph_[root].clear();
        nn_graph_[other].clear();

        // Also add bidirectional edges:
        for (auto const& [nn_new, new_dist] : nn_ij)
        {
            nn_graph_[nn_new].try_emplace(new_id, new_dist);
            min_dist_in_knn_[nn_new] = std::min(min_dist_in_knn_[nn_new], new_dist);
            min_dist_in_knn_[new_id] = std::min(min_dist_in_knn_[new_id], new_dist);
        }

        std::unordered_map<size_t, float> nn_ij_map(nn_ij.begin(), nn_ij.end());
        // Create new node with id 'new_id' and add its neighbours:
        nn_graph_[new_id] = nn_ij_map;

        return nn_ij_map;
    }

    std::vector<std::tuple<size_t, size_t, float>> incremental_nns::find_existing_contractions(const feature_index& index)
    {
        std::vector<std::tuple<size_t, size_t, float>> new_edges;
        const std::vector<faiss::Index::idx_t> active_nodes = index.get_active_nodes();
        if (active_nodes.size() == 1)
            return new_edges;

        // First check in existing NN graph for contraction edges:
        for (const auto i: active_nodes)
            for (auto const& [j, distance] : nn_graph_[i])
                if (index.node_active(j) && distance >= 0)
                    new_edges.push_back({i, j, distance});

        return new_edges;
    }

    std::vector<std::tuple<size_t, size_t, float>> incremental_nns::compute_new_contractions(const feature_index& index)
    {
        std::vector<std::tuple<size_t, size_t, float>> new_edges;
        const std::vector<faiss::Index::idx_t> active_nodes = index.get_active_nodes();
        if (active_nodes.size() == 1)
            return new_edges;
        const size_t eff_k = std::min(k_, active_nodes.size() - 1);
        const auto [nns, distances] = index.get_nearest_nodes(active_nodes, eff_k);
        new_edges.reserve(nns.size());
        insert_nn_to_graph(active_nodes, nns, distances, eff_k);

        size_t index_1d = 0;
        for (size_t idx = 0; idx != active_nodes.size(); ++idx)
        {
            const size_t i = active_nodes[idx];
            for (size_t i_n = 0; i_n != eff_k; ++i_n, ++index_1d)
            {
                const float current_distance = distances[index_1d];
                const size_t j = nns[index_1d];
                if (current_distance <= 0 || !index.node_active(j))
                    continue;

                new_edges.push_back({i, j, current_distance});
            }
        }
        return new_edges;
    }
}
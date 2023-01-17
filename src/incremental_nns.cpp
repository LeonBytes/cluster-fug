#include "incremental_nns.h"
#include "time_measure_util.h"
#include <limits>
#include <cassert>
#include <cstddef>
#include <algorithm>
#include <unordered_set>

namespace DENSE_MULTICUT {

    using idx_t = faiss::Index::idx_t;
    template<typename REAL>
    incremental_nns<REAL>::incremental_nns(
        const std::vector<idx_t>& query_nodes, const std::vector<idx_t>& nns, const std::vector<REAL>& nns_distances, const size_t n, const size_t k, const size_t k_inner)
    {
        // Store as undirected graph.
        nn_graph_ = std::vector<std::unordered_map<size_t, REAL>>(2 * n);
        nn_inverse_graph_ = std::vector<std::vector<size_t>>(2 * n);
        min_dist_in_knn_ = std::vector<REAL>(2 * n, std::numeric_limits<REAL>::infinity());
        k_ = k;
        k_inner_ = k_inner;
        insert_nn_to_graph(query_nodes, nns, nns_distances, k);
    }

    template<typename REAL>
    void incremental_nns<REAL>::insert_nn_to_graph(
        const std::vector<idx_t>& query_nodes, const std::vector<idx_t>& nns, const std::vector<REAL>& nns_distances, const size_t k)
    {
        nn_graph_ = std::vector<std::unordered_map<size_t, REAL>>(nn_graph_.size());
        nn_inverse_graph_ = std::vector<std::vector<size_t>>(nn_inverse_graph_.size());
        min_dist_in_knn_ = std::vector<REAL>(min_dist_in_knn_.size(), std::numeric_limits<REAL>::infinity());
        size_t index_1d = 0;
        for (size_t idx = 0; idx != query_nodes.size(); ++idx)
        {
            const size_t i = query_nodes[idx];
            for (size_t i_n = 0; i_n != k; ++i_n, ++index_1d)
            {
                const REAL current_distance = nns_distances[index_1d];
                if (current_distance < 0)
                    continue;

                const size_t j = nns[index_1d];
                nn_graph_[i].try_emplace(j, current_distance);
                nn_inverse_graph_[j].push_back(i);
                min_dist_in_knn_[i] = std::min(min_dist_in_knn_[i], current_distance);
            }
        }
    }

    template<typename REAL>
    std::vector<std::tuple<size_t, size_t, REAL>> incremental_nns<REAL>::merge_nodes(const size_t i, const size_t j, const size_t new_id, const feature_index<REAL>&index, const bool do_exhaustive_search)
    {
        const size_t root = nn_graph_[i].size() >= nn_graph_[j].size() ? i: j;
        const size_t other = root == i ? j : i;
        
        const size_t current_k = k_inner_;
        std::unordered_map<size_t, REAL> nn_ij;
        std::vector<std::tuple<size_t, size_t, REAL>> new_edges;

        const REAL upper_bound_outside_knn_ij = min_dist_in_knn_[i] + min_dist_in_knn_[j];

        REAL largest_distance = 0.0;

        auto check_insert_neighbour = [&](const size_t neighbour, const REAL distance) {
            if (distance >= upper_bound_outside_knn_ij)
            {
                largest_distance = std::max(largest_distance, distance);
                nn_ij.try_emplace(neighbour, distance);
                nn_inverse_graph_[neighbour].push_back(new_id);
                min_dist_in_knn_[new_id] = std::min(min_dist_in_knn_[new_id], distance);
                new_edges.push_back({new_id, neighbour, distance});
            }
        };

        // iterate over kNNs of root:
        for (auto const& [nn_root, cost_root] : nn_graph_[root])
        {
            if (nn_root == other || !index.node_active(nn_root))
                continue;
            // check if nn_root is also in kNN(other):
            const auto nn_other_iter = nn_graph_[other].find(nn_root);
            if (nn_other_iter != nn_graph_[other].end() && nn_ij.size() < current_k)
            {
                const REAL current_dist = cost_root + nn_other_iter->second;
                check_insert_neighbour(nn_root, current_dist);
            }
        }

        for (auto const& [nn_root, cost_root] : nn_graph_[root])
        {
            if (nn_root == other || !index.node_active(nn_root))
                continue;
            const auto nn_other_iter = nn_graph_[other].find(nn_root);
            if (nn_other_iter == nn_graph_[other].end() && nn_ij.size() < current_k)
            {
                // Compute cost between other and nn_root and add.
                const REAL new_dist = cost_root + index.inner_product(nn_root, other); 
                check_insert_neighbour(nn_root, new_dist);
            }
            // Remove root as neighbour of nn_root:
            nn_graph_[nn_root].erase(root);
        }

        // Now iterate over kNNs of other:
        for (auto const& [nn_other, cost_other] : nn_graph_[other])
        {
            if (nn_other == root || !index.node_active(nn_other))
                continue;
            // Skip if nn_other is also in kNN(root) as already considered above.
            if (nn_graph_[root].find(nn_other) == nn_graph_[root].end() && nn_ij.size() < current_k)
            {
                const REAL new_dist = cost_other + index.inner_product(nn_other, root); // Compute cost between root and nn_other and add.
                check_insert_neighbour(nn_other, new_dist);
            }
            // Remove other as neighbour of nn_other:
            nn_graph_[nn_other].erase(other);
        }
    
        // Create new node with id 'new_id' and add its neighbours:
        nn_graph_[new_id] = nn_ij;        
        
        // Now check all nodes which were using i, j as their NNs:
        std::unordered_set<idx_t> queries_for_nn, require_min_dist_update;
        auto check_inverse_nn_connections = [&](const size_t p, const size_t q)
        {
            for (const size_t p_user : nn_inverse_graph_[p])
            {
                if (p_user == q || !index.node_active(p_user))
                    continue;
                const REAL dist_p_new_id = index.inner_product(p_user, new_id);
                if (dist_p_new_id >= min_dist_in_knn_[p_user]) // new_id can directly be a NN of node which was using one of merging nodes.
                {
                    nn_graph_[p_user].try_emplace(new_id, dist_p_new_id);
                    nn_inverse_graph_[new_id].push_back(p_user);
                    new_edges.push_back({p_user, new_id, dist_p_new_id});
                    require_min_dist_update.insert(p_user);
                }
                else if(do_exhaustive_search)
                    queries_for_nn.insert(p_user);
                nn_graph_[p_user].erase(p);
            }
            nn_inverse_graph_[p].clear();
        };
        check_inverse_nn_connections(j, i);
        check_inverse_nn_connections(i, j);

        for(const auto node: require_min_dist_update)
        {
            min_dist_in_knn_[node] = std::numeric_limits<REAL>::infinity();
            for (auto const& [nn_node, cost] : nn_graph_[node])
            {
                if (!index.node_active(nn_node))
                    continue;
                min_dist_in_knn_[node] = std::min(min_dist_in_knn_[node], cost);
            }
        }

        if (do_exhaustive_search && (nn_ij.size() == 0 || largest_distance < upper_bound_outside_knn_ij || queries_for_nn.size() > 0) && index.nr_nodes() > 1)
        {
            if (nn_ij.size() == 0 || largest_distance < upper_bound_outside_knn_ij)
                queries_for_nn.insert(new_id);
            const auto num_nn = std::min(k_, index.nr_nodes() - 1);
            const std::vector<idx_t> query_vec(std::vector<idx_t>(queries_for_nn.begin(), queries_for_nn.end()));
            const auto [nns, distances] = index.get_nearest_nodes(query_vec, num_nn);
            size_t index_1d = 0;
            for (const auto q: query_vec)
            {
                min_dist_in_knn_[q] = std::numeric_limits<REAL>::infinity();
                for (int nn_idx = 0; nn_idx != num_nn; ++nn_idx, ++index_1d)
                {
                    const REAL current_distance = distances[index_1d];
                    if (current_distance >= 0.0)
                    {
                        nn_graph_[q].try_emplace(nns[index_1d], current_distance);
                        nn_inverse_graph_[nns[index_1d]].push_back(q);
                        min_dist_in_knn_[q] = std::min(min_dist_in_knn_[q], current_distance);
                        new_edges.push_back({q, nns[index_1d], current_distance});
                    }
                }
            }
        }
        
        // Remove root and other nodes.
        nn_graph_[i].clear();
        nn_graph_[j].clear();

        return new_edges;
    }

    template<typename REAL>
    std::vector<std::tuple<size_t, size_t, REAL>> incremental_nns<REAL>::find_existing_contractions(const feature_index<REAL>&index)
    {
        std::vector<std::tuple<size_t, size_t, REAL>> new_edges;
        const std::vector<idx_t> active_nodes = index.get_active_nodes();
        if (active_nodes.size() == 1)
            return new_edges;

        for (const auto i: active_nodes)
            for (auto const& [j, distance] : nn_graph_[i])
                if (index.node_active(j) && distance >= 0)
                    new_edges.push_back({i, j, distance});

        return new_edges;
    }

    template<typename REAL>
    void incremental_nns<REAL>::print(const feature_index<REAL>&index)
    {
        std::cout<<"incremental_nns active nodes:\n";
        const std::vector<idx_t> active_nodes = index.get_active_nodes();
        for (const auto i: active_nodes)
            for (auto const& [j, distance] : nn_graph_[i])
                if (index.node_active(j))
                    std::cout<<i<<" "<<j<<" "<<distance<<"\n";
    }
    template<typename REAL>
    std::vector<std::tuple<size_t, size_t, REAL>> incremental_nns<REAL>::compute_new_contractions(const feature_index<REAL>&index)
    {
        std::vector<std::tuple<size_t, size_t, REAL>> new_edges;
        const std::vector<idx_t> active_nodes = index.get_active_nodes();
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
                const REAL current_distance = distances[index_1d];
                const size_t j = nns[index_1d];
                assert(index.node_active(j));
                if (current_distance <= 0)
                    continue;

                assert(i != j);
                new_edges.push_back({i, j, current_distance});
            }
        }
        return new_edges;
    }
    
    template class incremental_nns<float>;
    template class incremental_nns<double>;
}
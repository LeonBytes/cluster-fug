#include "feature_index_brute_force.h"
#include "time_measure_util.h"
#include <cassert>
#include <queue>
//#include <iostream>

namespace DENSE_MULTICUT {

    template<typename REAL>
    std::tuple<idx_t, REAL> feature_index_brute_force<REAL>::get_nearest_node(const idx_t node) const 
    {
        const auto active_nodes = this->get_active_nodes();
        REAL best_distance = -1;
        idx_t best_id;
        for (const auto orig_node_n : active_nodes)
        {
            if (node == orig_node_n)
                continue;

            const auto sim = this->inner_product(node, orig_node_n);
            if (sim > best_distance)
            {
                best_distance = sim;
                best_id = orig_node_n;
            }
        }
        return {best_id, best_distance};
    }

    template<typename REAL>
    std::tuple<std::vector<idx_t>, std::vector<REAL>> feature_index_brute_force<REAL>::get_nearest_nodes(const std::vector<idx_t>& nodes) const
    {
        const auto num_query_nodes = nodes.size();
        std::vector<REAL> final_distances(num_query_nodes, -1.0);
        std::vector<idx_t> final_ids(num_query_nodes);
        const auto active_nodes = this->get_active_nodes();

        #pragma omp parallel for if (num_query_nodes > 100)
        for (size_t c = 0; c != num_query_nodes; ++c)
        {
            const auto orig_node_c = nodes[c];
            for (const auto orig_node_n : active_nodes)
            {
                if (orig_node_c == orig_node_n)
                    continue;

                const auto sim = this->inner_product(orig_node_c, orig_node_n);
                if (sim > final_distances[c])
                {
                    final_distances[c] = sim;
                    final_ids[c] = orig_node_n;
                }
            }
        }
        return {final_ids, final_distances};
    }

    template<typename REAL>
    std::tuple<std::vector<idx_t>, std::vector<REAL>> feature_index_brute_force<REAL>::get_nearest_nodes(const std::vector<idx_t>& nodes, const size_t k) const
    {
        const auto num_query_nodes = nodes.size();
        std::vector<REAL> final_distances(num_query_nodes * k, -1.0);
        std::vector<idx_t> final_ids(num_query_nodes * k);

        using pq_type = std::tuple<REAL, idx_t>;
        auto pq_comp = [](const pq_type& a, const pq_type& b) { return std::get<0>(a) > std::get<0>(b); };

        const auto active_nodes = this->get_active_nodes();
        #pragma omp parallel for if (num_query_nodes > 100)
        for (idx_t c = 0; c != num_query_nodes; ++c)
        {
            std::priority_queue<pq_type, std::vector<pq_type>, decltype(pq_comp)> pq(pq_comp);
            const auto orig_node_c = nodes[c];
            for (const auto orig_node_n : active_nodes)
            {
                if (orig_node_c == orig_node_n)
                    continue;

                const auto sim = this->inner_product(orig_node_c, orig_node_n);
                pq.push({sim, orig_node_n});
                if(pq.size() > k)
                    pq.pop();
            }
            for (int n_index = k - 1; n_index >= 0; --n_index)
            {
                const auto [distance, orig_node_n] = pq.top();
                pq.pop();
                final_distances[c * k + n_index] = distance;
                final_ids[c * k + n_index] = orig_node_n;
            }
        }
        return {final_ids, final_distances};
    }

    template class feature_index_brute_force<float>;
    template class feature_index_brute_force<double>;
}
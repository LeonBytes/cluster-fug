#include "feature_index.h"
#include "time_measure_util.h"
#include <faiss/index_factory.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/utils/distances.h>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include <queue>
//#include <iostream>

namespace DENSE_MULTICUT {

    template<typename REAL>        
    feature_index<REAL>::feature_index(const size_t _d, const size_t n, const std::vector<REAL>& _features, const bool track_dist_offset)
        : d(_d),
        features(_features),
        nr_active(n),
        track_dist_offset_(track_dist_offset)
    {
        active = std::vector<char>(n, true);
        vacant_node = n;
        internal_to_orig_node_mapping = std::vector<idx_t>(2 * n);
        orig_to_internal_node_mapping = std::vector<idx_t>(2 * n);
        std::iota(internal_to_orig_node_mapping.begin(), internal_to_orig_node_mapping.end(), 0);
        std::iota(orig_to_internal_node_mapping.begin(), orig_to_internal_node_mapping.end(), 0);
        mapping_is_identity = true;
    }

    template<typename REAL>
    void feature_index<REAL>::remove(const idx_t i_orig)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
        const idx_t i = get_orig_to_internal_node_mapping(i_orig);
        assert(i < active.size());
        assert(active[i] == true);
        active[i] = false;
        nr_active--;
    }

    template<typename REAL>
    idx_t feature_index<REAL>::merge(const idx_t i_orig, const idx_t j_orig, const bool add_to_index)
    {
        const idx_t i = get_orig_to_internal_node_mapping(i_orig);
        const idx_t j = get_orig_to_internal_node_mapping(j_orig);
        assert(i != j);
        assert(i < active.size());
        assert(j < active.size());

        active[i] = false;
        active[j] = false;

        nr_active--;

        const idx_t new_id = features.size()/d;
        for(size_t l=0; l<d; ++l)
            features.push_back(features[i*d + l] + features[j*d + l]);
        active.push_back(true);
        mapping_is_identity = false;
        internal_to_orig_node_mapping[new_id] = vacant_node;
        orig_to_internal_node_mapping[vacant_node] = new_id;
        return vacant_node++;
    }

    template<typename REAL>
    double feature_index<REAL>::inner_product(const idx_t i_orig, const idx_t j_orig) const
    {
        const idx_t i = get_orig_to_internal_node_mapping(i_orig);
        const idx_t j = get_orig_to_internal_node_mapping(j_orig);
        assert(i < active.size());
        assert(j < active.size());
        double x = 0.0;
        for(size_t l=0; l<d-1; ++l)
            x += features[i*d+l]*features[j*d+l];
        if(track_dist_offset_)
            x -= features[i*d+d-1]*features[j*d+d-1];
        else
            x += features[i*d+d-1]*features[j*d+d-1];
        return x;
    }

    template<typename REAL>
    bool feature_index<REAL>::node_active(const idx_t i_orig) const
    {
        const idx_t idx = get_orig_to_internal_node_mapping(i_orig);
        if (idx < active.size())
            return active[idx] == true;
        return false;
    }

    template<typename REAL>
    size_t feature_index<REAL>::max_id_nr() const 
    { 
        assert(active.size() > 0);
        return active.size()-1;
    }

    template<typename REAL>
    size_t feature_index<REAL>::nr_nodes() const
    {
        assert(nr_active == std::count(active.begin(), active.end(), true));
        return nr_active;
    }

    template<typename REAL>
    std::vector<idx_t> feature_index<REAL>::get_active_nodes() const
    {
        std::vector<idx_t> active_nodes;
        for (int i = 0; i != active.size(); ++i)
        {
            if (active[i])
            {
                active_nodes.push_back(get_internal_to_orig_node_mapping(i));
                assert(get_orig_to_internal_node_mapping(get_internal_to_orig_node_mapping(i)) == i);
            }
        }
        return active_nodes;
    }

    template<typename REAL>
    idx_t feature_index<REAL>::get_orig_to_internal_node_mapping(const idx_t i) const
    {
        if (mapping_is_identity)
            return i;
        return orig_to_internal_node_mapping[i];
    }

    template<typename REAL>
    idx_t feature_index<REAL>::get_internal_to_orig_node_mapping(const idx_t i) const
    {
        if (mapping_is_identity)
            return i;
        return internal_to_orig_node_mapping[i];
    }

    template <typename REAL>
    void feature_index<REAL>::reconstruct_clean_index(std::string new_index_str)
    {
        std::vector<REAL> active_features(nr_active * d);

        std::vector<idx_t> internal_to_orig_node_mapping_new(nr_active);
        int i_internal_new = 0;
        for (int i = 0; i != active.size(); ++i)
        {
            if (active[i])
            {
                const auto i_orig = get_internal_to_orig_node_mapping(i);
                orig_to_internal_node_mapping[i_orig] = i_internal_new;
                internal_to_orig_node_mapping_new[i_internal_new] = i_orig;
                std::copy(features.begin() + i * d, features.begin() + (i + 1) * d, active_features.begin() + (i_internal_new * d));
                active[i] = false;
                ++i_internal_new;
            }
        }
        internal_to_orig_node_mapping = internal_to_orig_node_mapping_new;
        std::swap(features, active_features);
        active.resize(nr_active);
        std::fill(active.begin(), active.end(), true);
        mapping_is_identity = false;
    }
    
    // template<typename REAL>
    // std::tuple<std::vector<idx_t>, std::vector<REAL>> feature_index<REAL>::get_nearest_nodes_brute_force(const std::vector<idx_t>& nodes) const
    // {
    //     const auto num_query_nodes = nodes.size();
    //     std::vector<REAL> final_distances(num_query_nodes, -1.0);
    //     std::vector<idx_t> final_ids(num_query_nodes);
    //     const auto active_nodes = get_active_nodes();

    //     #pragma omp parallel for if (num_query_nodes > 100)
    //     for (size_t c = 0; c != num_query_nodes; ++c)
    //     {
    //         const auto orig_node_c = nodes[c];
    //         for (const auto orig_node_n : active_nodes)
    //         {
    //             if (orig_node_c == orig_node_n)
    //                 continue;

    //             const auto sim = inner_product(orig_node_c, orig_node_n);
    //             if (sim > final_distances[c])
    //             {
    //                 final_distances[c] = sim;
    //                 final_ids[c] = orig_node_n;
    //             }
    //         }
    //     }
    //     return {final_ids, final_distances};
    // }

    // template<typename REAL>
    // std::tuple<std::vector<idx_t>, std::vector<REAL>> feature_index<REAL>::get_nearest_nodes_brute_force(const std::vector<idx_t>& nodes, const size_t k) const
    // {
    //     const auto num_query_nodes = nodes.size();
    //     std::vector<REAL> final_distances(num_query_nodes * k, -1.0);
    //     std::vector<idx_t> final_ids(num_query_nodes * k);

    //     using pq_type = std::tuple<REAL, idx_t>;
    //     auto pq_comp = [](const pq_type& a, const pq_type& b) { return std::get<0>(a) > std::get<0>(b); };

    //     const auto active_nodes = get_active_nodes();
    //     #pragma omp parallel for if (num_query_nodes > 100)
    //     for (idx_t c = 0; c != num_query_nodes; ++c)
    //     {
    //         std::priority_queue<pq_type, std::vector<pq_type>, decltype(pq_comp)> pq(pq_comp);
    //         const auto orig_node_c = nodes[c];
    //         for (const auto orig_node_n : active_nodes)
    //         {
    //             if (orig_node_c == orig_node_n)
    //                 continue;

    //             const auto sim = inner_product(orig_node_c, orig_node_n);
    //             pq.push({sim, orig_node_n});
    //             if(pq.size() > k)
    //                 pq.pop();
    //         }
    //         for (int n_index = k - 1; n_index >= 0; --n_index)
    //         {
    //             const auto [distance, orig_node_n] = pq.top();
    //             pq.pop();
    //             final_distances[c * k + n_index] = distance;
    //             final_ids[c * k + n_index] = orig_node_n;
    //         }
    //     }
    //     return {final_ids, final_distances};
    // }

    template class feature_index<float>;
    template class feature_index<double>;
}
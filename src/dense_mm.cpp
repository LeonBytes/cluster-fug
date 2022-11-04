#include "dense_mm.h"
#include "feature_index.h"
#include "dense_multicut_utils.h"
#include "union_find.hxx"
#include "time_measure_util.h"

#include <vector>
#include <queue>
#include <numeric>
#include <random>
#include <iostream>

#include <faiss/index_factory.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetaIndexes.h>

#include <faiss/IndexHNSW.h>

#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <faiss/impl/AuxIndexStructures.h>

namespace DENSE_MULTICUT {

    std::vector<size_t> dense_mm_impl(const size_t n, const size_t d, std::vector<float> features, const std::string index_str, const bool track_dist_offset)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        feature_index index(d, n, features, index_str, track_dist_offset);
        assert(features.size() == n*d);

        std::cout << "[dense mm " << index_str << "] Find multicut for " << n << " nodes with features of dimension " << d << "\n";

        double multicut_cost = cost_disconnected(n, d, features, track_dist_offset);

        const size_t max_nr_ids = 2*n;
        union_find uf(max_nr_ids);

        std::vector<char> forbidden_nodes(max_nr_ids, 0);
        std::vector<faiss::Index::idx_t> orig_node_ids(max_nr_ids);
        std::iota(orig_node_ids.begin(), orig_node_ids.end(), 0);

        using pq_type = std::tuple<float, std::array<faiss::Index::idx_t,2>>;
        auto pq_comp = [](const pq_type& a, const pq_type& b) { return std::get<0>(a) < std::get<0>(b); };
        std::priority_queue<pq_type, std::vector<pq_type>, decltype(pq_comp)> pq(pq_comp);

        {
            std::vector<faiss::Index::idx_t> all_indices(n);
            std::iota(all_indices.begin(), all_indices.end(), 0);
            const auto [nns, distances] = index.get_nearest_nodes(all_indices);
            std::cout<<"Initial NN search complete\n";
            for(size_t i=0; i<n; ++i)
            {
                if(distances[i] > 0.0)
                    pq.push({distances[i], {i,nns[i]}});
            }
        }

        bool terminate = false;
        size_t vacant_node_id = n;
        while (!terminate)
        {
            terminate = true;
            while(!pq.empty()) 
            {
                const auto [distance, ij] = pq.top();
                pq.pop();
                assert(distance > 0.0);
                const auto [i,j] = ij;
                assert(i != j);
                if(forbidden_nodes[i] || forbidden_nodes[j])
                    continue;

                forbidden_nodes[i] = 1;
                forbidden_nodes[j] = 1;
                const size_t new_id = index.merge(i, j); // new_id is computed w.r.t subset of nodes.
                assert(new_id == vacant_node_id);

                orig_node_ids[new_id] = orig_node_ids[vacant_node_id];
                uf.merge(orig_node_ids[i], vacant_node_id);
                uf.merge(orig_node_ids[j], vacant_node_id);

                vacant_node_id++;
                multicut_cost -= distance;
                terminate = false;
            }
            std::cout<<"Objective: "<<multicut_cost<<"\n";
            // Rebuild feature_index and insert NNs into PQ:
            std::vector<faiss::Index::idx_t> active_nodes;
            if (index.nr_nodes() < 0.8f * index.max_id_nr() && false)
            {
                const auto new_index_data = index.reconstruct_clean_index(orig_node_ids);
                const std::vector<float> new_features = std::get<0>(new_index_data);
                orig_node_ids = std::get<1>(new_index_data);
                std::cout<<"Objective: "<<multicut_cost<<". Reconstructing index with "<<index.nr_nodes()<<" nodes.\n";

                index = feature_index(d, index.nr_nodes(), new_features, index_str, track_dist_offset);
                active_nodes = std::vector<faiss::Index::idx_t>(index.nr_nodes());
                std::iota(active_nodes.begin(), active_nodes.end(), 0);
            }
            else
                active_nodes = index.get_active_nodes();
            std::fill(forbidden_nodes.begin(), forbidden_nodes.end(), 0);
            const auto [nns, distances] = index.get_nearest_nodes(active_nodes);
            for(const auto i: active_nodes)
            {
                assert(i < max_nr_ids);
                assert(nns[i] < max_nr_ids);
                if(distances[i] > 0.0)
                    pq.push({distances[i], {i,nns[i]}});
            }
        }
        std::cout << "[dense mm " << index_str << "] final nr clusters = " << uf.count() - (max_nr_ids - index.max_id_nr()-1) << "\n";
        std::cout << "[dense mm " << index_str << "] final multicut cost = " << multicut_cost << "\n";

        std::vector<size_t> component_labeling(n);
        for(size_t i=0; i<n; ++i)
            component_labeling[i] = uf.find(i);
        
        const double cost_computed = labeling_cost(component_labeling, n, d, features, track_dist_offset);
        std::cout << "[dense mm " << index_str << "] final multicut computed cost = " << cost_computed << "\n";
        return component_labeling;
    }

    std::vector<size_t> dense_mm_flat_index(const size_t n, const size_t d, std::vector<float> features, const bool track_dist_offset)
    {
        std::cout << "Dense MM with flat index\n";
        return dense_mm_impl(n, d, features, "Flat", track_dist_offset);
    }

    std::vector<size_t> dense_mm_hnsw(const size_t n, const size_t d, std::vector<float> features, const bool track_dist_offset)
    {
        std::cout << "Dense MM with HNSW index\n";
        return dense_mm_impl(n, d, features, "HNSW", track_dist_offset);
    }

}

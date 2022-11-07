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
                const size_t new_id = index.merge(i, j);

                uf.merge(i, new_id);
                uf.merge(j, new_id);

                multicut_cost -= distance;
                terminate = false;
            }
            // Rebuild feature_index and insert NNs into PQ:
            std::vector<faiss::Index::idx_t> active_nodes;
            if (index.nr_nodes() < 0.9f * index.max_id_nr() && index.nr_nodes() > 1000)
            {
                index.reconstruct_clean_index();
                std::cout<<"Objective: "<<multicut_cost<<". Reconstructed index with "<<index.nr_nodes()<<" nodes.\n";
            }
            else
                active_nodes = index.get_active_nodes();
            const auto [nns, distances] = index.get_nearest_nodes(active_nodes);
            for(size_t idx = 0; idx != active_nodes.size(); ++idx)
            {
                const auto i = active_nodes[idx];
                if(distances[idx] > 0.0)
                {
                    const auto j = nns[idx];
                    pq.push({distances[idx], {i, j}});
                    forbidden_nodes[i] = 0;
                    forbidden_nodes[j] = 0;
                }
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

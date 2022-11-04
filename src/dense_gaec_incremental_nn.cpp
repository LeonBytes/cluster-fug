#include "dense_gaec_incremental_nn.h"
#include "feature_index.h"
#include "incremental_nns.h"
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

    using pq_type = std::tuple<float, std::array<faiss::Index::idx_t,2>>;
    template<typename T, class Container=std::vector<T>, class Compare=std::less<typename Container::value_type>> 
    class priority_queue_with_deletion : public std::priority_queue<T, Container, Compare>
    {
        public:
            // inherit constructors
            using std::priority_queue<T, Container, Compare>::priority_queue;
            void remove_invalid(const feature_index& index) {
                std::vector<pq_type> retained_edges;
                retained_edges.reserve(this->size());
                for (auto it = this->c.begin(); it != this->c.end();++it)
                {
                    const auto [i,j] = std::get<1>(*it);
                    // check if edge is still present in contracted graph. This is true if both endpoints have not been contracted
                    if(index.node_active(i) && index.node_active(j))
                        retained_edges.push_back(*it);
                }
                this->c = retained_edges;
                std::make_heap(this->c.begin(), this->c.end(), this->comp);
            }
    };

    std::vector<size_t> dense_gaec_incremental_nn(const size_t n, const size_t d, std::vector<float> features, const size_t k_in, const std::string index_str, const bool track_dist_offset)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        const size_t k = std::min(n - 1, k_in);
        feature_index index(d, n, features, index_str, track_dist_offset);
        assert(features.size() == n*d);

        std::cout << "[dense gaec incremental nn] Find multicut for " << n << " nodes with features of dimension " << d << " and feature index type "<<index_str<<"\n";

        double multicut_cost = cost_disconnected(n, d, features, track_dist_offset);

        const size_t max_nr_ids = 2*n;
        union_find uf(max_nr_ids);

        incremental_nns nn_graph;
        auto pq_comp = [](const pq_type& a, const pq_type& b) { return std::get<0>(a) < std::get<0>(b); };
        priority_queue_with_deletion<pq_type, std::vector<pq_type>, decltype(pq_comp)> pq(pq_comp);
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("Initial KNN construction");
            std::vector<faiss::Index::idx_t> all_indices(n);
            std::iota(all_indices.begin(), all_indices.end(), 0);
            const auto [nns, distances] = index.get_nearest_nodes(all_indices, k);
            std::cout<<"[dense gaec incremental nn] Initial NN search complete\n";
            nn_graph = incremental_nns(all_indices, nns, distances, n, k);
            size_t index_1d = 0;
            for(size_t i=0; i<n; ++i)
                for(size_t i_k=0; i_k < k; ++i_k, ++index_1d)
                    if(distances[index_1d] > 0.0)
                        pq.push({distances[index_1d], {i, nns[index_1d]}});
        }
        const size_t max_pq_size = pq.size() * 10;

        // iteratively find pairs of features with highest inner product
        bool completed = false;
        while(!pq.empty() || !completed) {
            if (pq.size() == 0)
            {
                const std::vector<std::tuple<size_t, size_t, float>> remaining_edges = nn_graph.recheck_possible_contractions(index);
                for (const auto [i, j, cost]: remaining_edges)
                    pq.push({cost, {i, j}});
                std::cout<<"Found "<<pq.size()<<" leftover contractions.\n";
                completed = pq.size() == 0;
                continue;
            }
            const auto [distance, ij] = pq.top();
            pq.pop();
            assert(distance >= 0.0);
            const auto [i,j] = ij;
            assert(i != j);
            // check if edge is still present in contracted graph. This is true if both endpoints have not been contracted
            if(index.node_active(i) && index.node_active(j))
            {
                // std::cout << "[dense gaec incremental nn] contracting edge " << i << " and " << j << " with edge cost " << distance << "\n";
                // contract edge:
                const size_t new_id = index.merge(i,j);

                uf.merge(i, new_id);
                uf.merge(j, new_id);
                const std::unordered_map<size_t, float> nn_ij = nn_graph.merge_nodes(i, j, new_id, index);
                multicut_cost -= distance;
                // find new nearest neighbor
                if(index.nr_nodes() > 1)
                    for (auto const& [nn_new, new_cost] : nn_ij)
                        pq.push({new_cost, {new_id, nn_new}});
            }
            if (pq.size() > max_pq_size)
            {
                std::cout<<"[dense gaec incremental nn] cleaning-up PQ with size: "<<pq.size();
                pq.remove_invalid(index);
                std::cout<<", new PQ size: "<<pq.size()<<"\n";
            }
        }

        std::cout << "[dense gaec incremental nn] final nr clusters = " << uf.count() - (max_nr_ids - index.max_id_nr()-1) << "\n";
        std::cout << "[dense gaec incremental nn] final multicut cost = " << multicut_cost << "\n";

        std::vector<size_t> component_labeling(n);
        for(size_t i=0; i<n; ++i)
            component_labeling[i] = uf.find(i);
        return component_labeling;
    }
}


// TODO:
// 1. Remove features of inactive nodes.
// 2. Reinitialize feature_index upon termination to check if all costs still < 0.
// 3. 2nd element of PQ can give a cost value, stating only find NN above this cost. As otherwise newly added edge is not going to be contracted now anyway.


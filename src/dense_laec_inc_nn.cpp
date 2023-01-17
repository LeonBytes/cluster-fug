#include "dense_laec_inc_nn.h"
#include "feature_index.h"
#include "feature_index_faiss.h"
#include "feature_index_brute_force.h"
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
            void remove_invalid(const feature_index<T>& index) {
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

    template<typename REAL>
    std::vector<size_t> dense_laec_inc_nn_impl(
        const size_t n, const size_t d, feature_index<REAL>& index, const std::vector<REAL>& features, const size_t k_in, 
        const bool track_dist_offset, const std::string index_str_after, const size_t k_cap)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
        const size_t k = std::min(n - 1, k_in);
        const size_t k_cap_eff = std::min(n - 1, k_cap);
        
        assert(features.size() == n*d);

        std::cout << "[dense laec] Find multicut for " << n << " nodes with features of dimension " << d << " with k " <<k<<" and K "<<k_cap_eff<<"\n";
        double multicut_cost = cost_disconnected(n, d, features, track_dist_offset);

        const size_t max_nr_ids = 2*n;
        union_find uf(max_nr_ids);

        incremental_nns<REAL> nn_graph;
        auto pq_comp = [](const pq_type& a, const pq_type& b) { return std::get<0>(a) < std::get<0>(b); };
        priority_queue_with_deletion<pq_type, std::vector<pq_type>, decltype(pq_comp)> pq(pq_comp);
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("Initial KNN construction");
            std::vector<faiss::Index::idx_t> all_indices(n);
            std::iota(all_indices.begin(), all_indices.end(), 0);
            const auto [nns, distances] = index.get_nearest_nodes(all_indices, k);
            std::cout<<"[dense laec] Initial NN search complete\n";
            nn_graph = incremental_nns(all_indices, nns, distances, n, k, k_cap_eff);
            size_t index_1d = 0;
            for(size_t i=0; i<n; ++i)
                for(size_t i_k=0; i_k < k; ++i_k, ++index_1d)
                    if(distances[index_1d] > 0.0)
                        pq.push({distances[index_1d], {i, nns[index_1d]}});
        }
        auto insert_into_pq = [&](const std::vector<std::tuple<size_t, size_t, REAL>>& edges) {
            for (const auto [i, j, cost]: edges)
            {
                assert(index.node_active(i));
                assert(index.node_active(j));
                pq.push({cost, {i, j}});
            }
        };

        const size_t max_pq_size = pq.size() * 10;

        size_t current_pq_size = pq.size();
        bool completed = false;
        while (!completed)
        {        
            while(!pq.empty()) 
            {
                const auto [distance, ij] = pq.top();
                pq.pop();
                assert(distance >= 0.0);
                const auto [i,j] = ij;
                assert(i != j);
                if(index.node_active(i) && index.node_active(j))
                {
                    const size_t new_id = index.merge(i, j, false); // Do not add to index, since new index is going to be built anyway.

                    uf.merge(i, new_id);
                    uf.merge(j, new_id);
                    insert_into_pq(nn_graph.merge_nodes(i, j, new_id, index, false));
                    multicut_cost -= distance;
                    completed = false;
                }
                // if (pq.size() > max_pq_size)
                // {
                //     std::cout<<"[dense laec] cleaning-up PQ with size: "<<pq.size();
                //     pq.remove_invalid(index);
                //     std::cout<<", new PQ size: "<<pq.size()<<"\n";
                // }
            }

            // Rebuild feature_index and insert NNs into PQ:
            std::cout<<"Objective: "<<multicut_cost<<". Number of clusters: "<<index.nr_nodes()<<".\n";
            // insert_into_pq(nn_graph.find_existing_contractions(index));
            // if ((index.nr_nodes() < 0.9f * index.max_id_nr() && index.nr_nodes() > 1000) || pq.size() == 0)
            if (pq.size() == 0)
            {
                index.reconstruct_clean_index(index_str_after);
                std::cout<<"Reconstructed index of type "<<index_str_after<<" with "<<index.nr_nodes()<<" nodes.\n";
                insert_into_pq(nn_graph.compute_new_contractions(index));
            }

            std::cout<<"Found "<<pq.size()<<" leftover contractions.\n";
            completed = pq.size() == 0;
        }

        std::cout << "[dense laec] final nr clusters = " << index.nr_nodes() << "\n";
        std::cout << "[dense laec] final multicut cost = " << multicut_cost << "\n";

        std::vector<size_t> component_labeling(n);
        for(size_t i=0; i<n; ++i)
            component_labeling[i] = uf.find(i);

        // std::cout << "[dense laec] final multicut computed cost = " << labeling_cost(component_labeling, n, d, features, track_dist_offset) << "\n";
        return component_labeling;
    }

    std::vector<size_t> dense_laec_inc_nn_faiss(
        const size_t n, const size_t d, const std::vector<float>& features, const std::string index_str, const bool track_dist_offset, const size_t k_in, const size_t k_cap)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
        std::cout << "Dense LAEC with faiss index: "<<index_str<<"\n";
        std::unique_ptr<feature_index_faiss> index = std::make_unique<feature_index_faiss>(d, n, features, index_str, track_dist_offset);
        return dense_laec_inc_nn_impl<float>(n, d, *index, features, k_in, track_dist_offset, index_str, k_cap);
    }

    std::vector<size_t> dense_laec_inc_nn_faiss_bf_after(
        const size_t n, const size_t d, const std::vector<float>& features, const std::string index_str, const bool track_dist_offset, const size_t k_in, const size_t k_cap)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
        std::cout << "Dense LAEC with faiss index: "<<index_str<<", and faiss_brute_force for later iterations\n";
        std::unique_ptr<feature_index_faiss> index = std::make_unique<feature_index_faiss>(d, n, features, index_str, track_dist_offset);
        return dense_laec_inc_nn_impl<float>(n, d, *index, features, k_in, track_dist_offset, "faiss_brute_force", k_cap);
    }

    template<typename REAL>
    std::vector<size_t> dense_laec_inc_nn_brute_force(
        const size_t n, const size_t d, const std::vector<REAL>& features, const bool track_dist_offset, const size_t k_in, const size_t k_cap)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
        std::cout << "Dense LAEC with brute force\n";
        std::unique_ptr<feature_index_brute_force<REAL>> index = std::make_unique<feature_index_brute_force<REAL>>(
                                                                    d, n, features, track_dist_offset);
        return dense_laec_inc_nn_impl<REAL>(n, d, *index, features, k_in, track_dist_offset, "brute_force", k_cap);
    }

    template std::vector<size_t> dense_laec_inc_nn_brute_force(const size_t, const size_t, const std::vector<float>&, const bool, const size_t, const size_t);
    template std::vector<size_t> dense_laec_inc_nn_brute_force(const size_t, const size_t, const std::vector<double>&, const bool, const size_t, const size_t);
}
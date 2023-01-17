#include "dense_gaec.h"
#include "feature_index.h"
#include "feature_index_faiss.h"
#include "feature_index_brute_force.h"
#include "dense_multicut_utils.h"
#include "union_find.hxx"
#include "time_measure_util.h"
#include "incremental_nns.h"

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

    template<typename REAL>
    std::vector<size_t> dense_gaec_impl(const size_t n, const size_t d, feature_index<REAL>& index, std::vector<REAL> features, const bool track_dist_offset)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;

        assert(features.size() == n*d);

        std::cout << "[dense gaec] Find multicut for " << n << " nodes with features of dimension " << d << "\n";

        double multicut_cost = cost_disconnected(n, d, features, track_dist_offset);

        const size_t max_nr_ids = 2*n;
        union_find uf(max_nr_ids);
        // incremental_nns<REAL> nn_graph;

        using pq_type = std::tuple<REAL, std::array<faiss::Index::idx_t,2>>;
        auto pq_comp = [](const pq_type& a, const pq_type& b) { return std::get<0>(a) < std::get<0>(b); };
        std::priority_queue<pq_type, std::vector<pq_type>, decltype(pq_comp)> pq(pq_comp);
        std::vector<std::vector<std::pair<u_int32_t, REAL>>> pq_pair(max_nr_ids);

        {
            std::vector<faiss::Index::idx_t> all_indices(n);
            std::iota(all_indices.begin(), all_indices.end(), 0);
            const auto [nns, distances] = index.get_nearest_nodes(all_indices);
            // nn_graph = incremental_nns(all_indices, nns, distances, n, 1);
            for(size_t i=0; i<n; ++i)
            {
                if(distances[i] > 0.0)
                {
                    pq.push({distances[i], {i,nns[i]}});
                    pq_pair[nns[i]].push_back({i, distances[i]});
                    //std::cout << "[dense gaec] push initial shortest edge " << i << " <-> " << nns[i] << " with cost " << distances[i] << "\n";
                }
            }
        }
        //std::cout << "[dense gaec] Added " << pq.size() << " initial elements to priority queue\n";

        // iteratively find pairs of features with highest inner product
        while(!pq.empty()) {
            const auto [distance, ij] = pq.top();
            pq.pop();
            assert(distance > 0.0);
            const auto [i,j] = ij;
            assert(i != j);
            // check if edge is still present in contracted graph. This is true if both endpoints have not been contracted
            if(index.node_active(i) && index.node_active(j))
            {
                //std::cout << "[dense multicut] contracting edge " << i << " and " << j << " with edge cost " << distance << "\n";
                // contract edge:
                const size_t new_id = index.merge(i,j);

                uf.merge(i, new_id);
                uf.merge(j, new_id);

                multicut_cost -= distance;

                // find new nearest neighbor
                if(index.nr_nodes() > 1)
                {
                    // std::unordered_map<size_t, REAL> nn_ij = nn_graph.merge_nodes(i, j, new_id, index, false);
                    // for (auto const& [nn_new, new_cost] : nn_ij)
                    // {
                    //     pq.push({new_cost, {new_id, nn_new}});
                    //     pq_pair[nn_new].push_back({new_id, new_cost});
                    // }
                    const bool new_id_nn_found = false; //nn_ij.size() > 0;
                    std::vector<faiss::Index::idx_t> new_query;
                    if (!new_id_nn_found)
                        new_query.push_back(new_id);

                    for(const auto k : pq_pair[i])
                    {
                        if(index.node_active(k.first))
                        {
                            const auto ip = index.inner_product(k.first, new_id);
                            if (ip > k.second)
                            {
                                pq.push({ip, {new_id, k.first}});
                                pq_pair[new_id].push_back({k.first, ip});
                            }
                            else
                                new_query.push_back(k.first);
                        }
                    }
                    for(const auto k : pq_pair[j])
                    {
                        if(index.node_active(k.first))
                        {
                            const auto ip = index.inner_product(k.first, new_id);
                            if (ip > k.second)
                            {
                                pq.push({ip, {new_id, k.first}});
                                pq_pair[new_id].push_back({k.first, ip});
                            }
                            else
                                new_query.push_back(k.first);
                        }
                    }
                    pq_pair[i].clear();
                    pq_pair[j].clear();

                    const auto [new_nns, new_distances] = index.get_nearest_nodes(new_query);
                    for(size_t c=0; c<new_nns.size(); ++c)
                    {
                        if(new_distances[c] > 0.0)
                        {
                            pq.push({new_distances[c], {new_nns[c], new_query[c]}});
                            pq_pair[new_nns[c]].push_back({new_query[c], new_distances[c]});
                            // if (!new_id_nn_found && new_query[c] == new_id)
                            //     nn_ij.try_emplace(new_nns[c], new_distances[c]);
                        }
                    }
                    // if(!new_id_nn_found)
                    //     nn_graph.update_graph(i, j, new_id, nn_ij);
                }
            }
        }

        std::cout << "[dense gaec] final nr clusters = " << uf.count() - (max_nr_ids - index.max_id_nr()-1) << "\n";
        std::cout << "[dense gaec] final multicut cost = " << multicut_cost << "\n";

        std::vector<size_t> component_labeling(n);
        for(size_t i=0; i<n; ++i)
            component_labeling[i] = uf.find(i);
        return component_labeling;
    }

    std::vector<size_t> dense_gaec_faiss(const size_t n, const size_t d, std::vector<float> features, const std::string index_str, const bool track_dist_offset)
    {
        std::cout << "Dense GAEC with faiss index: "<<index_str<<"\n";
        std::unique_ptr<feature_index_faiss> index = std::make_unique<feature_index_faiss>(d, n, features, index_str, track_dist_offset);
        return dense_gaec_impl<float>(n, d, *index, features, track_dist_offset);
    }

    template<typename REAL>
    std::vector<size_t> dense_gaec_brute_force(const size_t n, const size_t d, std::vector<REAL> features, const bool track_dist_offset)
    {
        std::cout << "Dense GAEC with brute force\n";
        std::unique_ptr<feature_index_brute_force<REAL>> index = std::make_unique<feature_index_brute_force<REAL>>(
                                                                    d, n, features, track_dist_offset);
        return dense_gaec_impl<REAL>(n, d, *index, features, track_dist_offset);
    }

    template std::vector<size_t> dense_gaec_brute_force(const size_t n, const size_t d, std::vector<float> features, const bool track_dist_offset);
    template std::vector<size_t> dense_gaec_brute_force(const size_t n, const size_t d, std::vector<double> features, const bool track_dist_offset);
}

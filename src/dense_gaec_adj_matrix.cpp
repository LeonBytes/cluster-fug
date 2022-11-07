#include "dense_gaec_adj_matrix.h"
#include "dense_multicut_utils.h"
#include "union_find.hxx"
#include "time_measure_util.h"

#include <iostream>
#include <vector>
#include <queue>
#include <cassert>
#include <functional>

namespace DENSE_MULTICUT {

    std::vector<size_t> dense_gaec_adj_matrix(const size_t n, const size_t d, std::vector<float> features, const bool track_dist_offset)
    {
        MEASURE_FUNCTION_EXECUTION_TIME;
        std::cout << "[dense gaec adj matrix] compute multicut on graph with " << n << " nodes with " << d << " feature dimensions\n";
        double multicut_cost = cost_disconnected(n, d, features, track_dist_offset);

        //std::vector<std::tuple<float,u_int32_t>> edges(((n-1)*n)/2, 0.0);
        std::vector<std::tuple<float,u_int32_t>> edges(n*n, {0.0, 0});

        auto edge_cost = [&](size_t i, size_t j) -> float& {
            assert(i != j);
            assert(i < n && j < n);
            if(i>j)
                std::swap(i,j);
            //const size_t idx = i*(i-1)/2 + j-1;
            const size_t idx = i*n+j;
            assert(idx < edges.size());
            return std::get<0>(edges[idx]);
        };

        auto edge_stamp = [&](u_int32_t i, u_int32_t j) -> u_int32_t& {
            assert(i != j);
            assert(i < n && j < n);
            if(i>j)
                std::swap(i,j);
            //const size_t idx = i*(i-1)/2 + j-1;
            const size_t idx = i*n+j;
            assert(idx < edges.size());
            return std::get<1>(edges[idx]);
        };

        auto inner_prod = [&](const u_int32_t i, const u_int32_t j) {
            assert(i != j);
            assert(i < n && j < n);
            float s = 0.0;
            for(size_t l=0; l<d-1; ++l)
                s += features[i*d+l] * features[j*d+l];
            if (track_dist_offset)
                s -= features[i*d+d-1] * features[j*d+d-1];
            else
                s += features[i*d+d-1] * features[j*d+d-1];
            return s;
        };

        for(u_int32_t i=0; i<n; ++i)
            for(u_int32_t j=0; j<i; ++j)
            {
                edge_cost(i,j) = inner_prod(i,j);
                //std::cout << "[dense multicut adjacency matrix] inner prod between " << i << " and " << j << " = " << inner_prod(i,j) << " = " << edge_cost(i,j) << "\n";
            }

        struct edge_type_q : public std::array<u_int32_t,2> {    
            float cost;    
            u_int32_t stamp;    
        };    

        auto pq_cmp = [](const edge_type_q& e1, const edge_type_q& e2) { return e1.cost < e2.cost; };    
        std::priority_queue<edge_type_q, std::vector<edge_type_q>, decltype(pq_cmp)> pq(pq_cmp);    

        for(u_int32_t i=0; i<n; ++i)
            for(u_int32_t j=0; j<i; ++j)
                if(edge_cost(i,j) > 0.0)
                {
                    pq.push(edge_type_q{i, j, edge_cost(i,j), 0});    
                    //std::cout << "[dense gaec adjacency matrix] push initial shortest edge " << i << " <-> " << j << " with cost " << edge_cost(i,j) << "\n";
                }

        std::vector<char> active(n, true);
        union_find uf(n);

        while(!pq.empty())
        {
            const edge_type_q e_q = pq.top();    
            pq.pop();    
            const u_int32_t i = e_q[0];    
            const u_int32_t j = e_q[1];
            assert(i != j && i < n && j < n);

            if(e_q.stamp < edge_stamp(i,j) || active[i] == false | active[j] == false)
                continue;


            uf.merge(i,j);
            multicut_cost -= edge_cost(i,j);
            std::cout << "[dense multicut adjacency matrix] contracting edge " << i << " and " << j << " with edge cost " << edge_cost(i,j) << ", multicut_cost "<< multicut_cost <<", stamp "<< edge_stamp(i,j) <<"\n";
            active[j] = false;

            // contract edge
 
            // update feature
            //for(size_t l=0; l<d; ++l)
            //    features[i*d+l] = features[i*d+l] + features[j*d+l];

            for(u_int32_t k=0; k<n; ++k)
            {
                if(i != k && j != k && active[k])
                {
                    edge_cost(i,k) = edge_cost(i,k) + edge_cost(j,k);
                    edge_stamp(i,k)++;

                    if(edge_cost(i,k) > 0.0)
                        pq.push(edge_type_q{i, k, edge_cost(i,k), edge_stamp(i,k)});
                }
            }
        }

        std::cout << "[dense gaec adj matrix] final nr clusters = " << uf.count() << "\n";
        std::cout << "[dense gaec adj matrix] final multicut cost = " << multicut_cost << "\n";

        std::vector<size_t> cc_ids(n);
        for(size_t i=0; i<n; ++i)
            cc_ids[i] = uf.find(i);

        // std::cout << "[dense gaec incremental nn] final multicut computed cost = " << labeling_cost(cc_ids, n, d, features, track_dist_offset) << "\n";

        return cc_ids; 
    }

}

#include "feature_index.h"
#include "feature_index_faiss.h"
#include "time_measure_util.h"
#include <faiss/index_factory.h>
#ifdef FAISS_ENABLE_GPU
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuCloner.h>
#endif
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/utils/distances.h>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <unordered_map>

namespace DENSE_MULTICUT {

    IDSelectorCustom::IDSelectorCustom(size_t n) : n(n), active(std::vector<char>(n, true)) {}
    bool IDSelectorCustom::is_member(idx_t id) const { return active[id] == true; }
    void IDSelectorCustom::merge(const idx_t i, const idx_t j) 
    {
        active[i] = false;
        active[j] = false;
        active.push_back(true);
        ++n;
    }

    feature_index_faiss::feature_index_faiss(const size_t _d, const size_t n, const std::vector<float>& _features, const std::string& _index_str, const bool track_dist_offset)
        : feature_index<float>(_d, n, _features, track_dist_offset),
        index_str(_index_str)
    {
        if (index_str != "faiss_brute_force")
        {
            index = std::shared_ptr<faiss::Index>(faiss::index_factory(d, index_str.c_str(), faiss::MetricType::METRIC_INNER_PRODUCT));
            #ifdef FAISS_ENABLE_GPU
                try
                {
                    int ngpus = faiss::gpu::getNumDevices();
                    printf("Number of GPUs: %d\n", ngpus);
                    if (ngpus > 0 && _index_str == "Flat")
                    {
                        faiss::gpu::StandardGpuResources res;
                        // make it into a gpu index
                        index.reset(faiss::gpu::index_cpu_to_gpu(&res, 0, index.get()));
                        std::cout<<"Using GPU index.\n";
                    }
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                    std::cout<<"Not using GPU index\n.";
                }            
            #endif

            index->train(n, features.data());
            {
                MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("faiss add");
                index->add(n, features.data());
            }
        }
        id_selector = new IDSelectorCustom(n);
    }


    idx_t feature_index_faiss::merge(const idx_t i_orig, const idx_t j_orig, const bool add_to_index)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
        const faiss::Index::idx_t new_internal_id = features.size()/d;
        const idx_t new_orig_id = feature_index<float>::merge(i_orig, j_orig);
        if (add_to_index && index_str != "faiss_brute_force")
            index->add(1, features.data() + new_internal_id * d);
        id_selector->merge(this->get_orig_to_internal_node_mapping(i_orig), this->get_orig_to_internal_node_mapping(j_orig));
        return new_orig_id;
    }

    void feature_index_faiss::reconstruct_clean_index(std::string new_index_str)
    {
        feature_index<float>::reconstruct_clean_index();
        if (new_index_str == "")
            new_index_str = index_str;
        index_str = new_index_str;
        if (index_str != "faiss_brute_force")
        {
            index = std::shared_ptr<faiss::Index>(faiss::index_factory(d, index_str.c_str(), faiss::MetricType::METRIC_INNER_PRODUCT));
            index->train(nr_active, features.data());
            index->add(nr_active, features.data());
        }
        delete id_selector;
        id_selector = new IDSelectorCustom(nr_active);
    }

    std::tuple<idx_t, float> feature_index_faiss::get_nearest_node(const idx_t id_orig) const
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
        if (index_str == "faiss_brute_force")
        {
            const std::tuple<std::vector<idx_t>, std::vector<float>> result = get_nearest_nodes_brute_force({id_orig}, 1);
            return {std::get<0>(result)[0], std::get<1>(result)[0]};
        }

        const size_t id = this->get_orig_to_internal_node_mapping(id_orig);
        std::vector<float> query_features(features.begin() + id * d, features.begin() + (id + 1) * d);
        if (track_dist_offset_)
            query_features[d - 1] *= -1.0;
        idx_t nearest_node = -1;
        float nearest_distance = 0.0; 
        for (size_t nr_lookups = 2; nr_lookups < 2 * index->ntotal && nearest_node == -1; nr_lookups *= 2)
        {
            float distance[std::min(nr_lookups, size_t(index->ntotal))];
            idx_t nns[std::min(nr_lookups, size_t(index->ntotal))];
            index->search(1, query_features.data() + id * d, std::min(nr_lookups, size_t(index->ntotal)), distance, nns);
            assert(std::is_sorted(distance, distance + std::min(nr_lookups, size_t(index->ntotal)), std::greater<float>()));
            for (size_t k = 0; k < std::min(nr_lookups, size_t(index->ntotal)); ++k)
                if (nns[k] < active.size() && nns[k] != id && active[nns[k]] == true)
                {
                    nearest_node = this->get_internal_to_orig_node_mapping(nns[k]);
                    nearest_distance = distance[k];
                    break;
                }
        }
        if (nearest_node == -1)
            throw std::runtime_error("Could not find nearest neighbor");

        return {nearest_node, nearest_distance};
    }

    std::tuple<std::vector<idx_t>, std::vector<float>> feature_index_faiss::get_nearest_nodes(const std::vector<idx_t> &nodes_orig) const
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("faiss get nearest nodes");
        assert(nodes_orig.size() > 0);
        if (index_str == "faiss_brute_force")
            return get_nearest_nodes_brute_force(nodes_orig, 1);
        //std::cout << "[feature index] search nearest neighbors for " << nodes.size() << " nodes\n";

        std::vector<idx_t> return_nns(nodes_orig.size());
        std::vector<float> return_distances(nodes_orig.size());

        std::unordered_map<idx_t, idx_t> node_map;
        node_map.reserve(nodes_orig.size());
        for (size_t c = 0; c < nodes_orig.size(); ++c)
            node_map.insert({this->get_orig_to_internal_node_mapping(nodes_orig[c]), c});

        for (size_t _nr_lookups = 2; _nr_lookups < 2 + 2 * max_id_nr(); _nr_lookups *= 2)
        {
            const size_t nr_lookups = std::min(_nr_lookups, size_t(index->ntotal));
            if (node_map.size() > 0)
            {
                // std::cout << "[feature index get_nearest_nodes] nr lookups = " << nr_lookups << ", num nodes = "<<node_map.size()<<"\n";
                std::vector<idx_t> cur_nodes;
                for (const auto [node, idx] : node_map)
                    cur_nodes.push_back(node);
                std::vector<idx_t> nns(cur_nodes.size() * nr_lookups);
                std::vector<float> distances(cur_nodes.size() * nr_lookups);

                std::vector<float> query_features(node_map.size() * d);
                for (size_t c = 0; c < cur_nodes.size(); ++c)
                {
                    for (size_t l = 0; l < d; ++l)
                        query_features[c * d + l] = features[cur_nodes[c] * d + l];
                    if (track_dist_offset_)
                        query_features[c * d + d - 1] *= -1.0;
                }
                {
                    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("faiss search");
                    index->search(cur_nodes.size(), query_features.data(), nr_lookups, distances.data(), nns.data());
                }

                for (size_t c = 0; c < cur_nodes.size(); ++c)
                {
                    for (size_t k = 0; k < nr_lookups; ++k)
                    {
                        if (nns[c * nr_lookups + k] < active.size() && nns[c * nr_lookups + k] != cur_nodes[c] && active[nns[nr_lookups * c + k]] == true)
                        {
                            assert(node_map.count(cur_nodes[c]) > 0);
                            return_nns[node_map[cur_nodes[c]]] = this->get_internal_to_orig_node_mapping(nns[c * nr_lookups + k]);
                            return_distances[node_map[cur_nodes[c]]] = distances[c * nr_lookups + k];
                            node_map.erase(cur_nodes[c]);
                            break;
                        }
                    }
                }
            }
        }

            for(size_t i=0; i<return_nns.size(); ++i)
                assert(return_nns[i] != nodes_orig[i]);
            return {return_nns, return_distances};
    }

    std::tuple<std::vector<idx_t>, std::vector<float>> feature_index_faiss::get_nearest_nodes(const std::vector<idx_t>& nodes_orig, const size_t k) const
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("faiss get k nearest nodes");
        assert(k > 0);
        assert(k < nr_nodes());
        assert(nodes_orig.size() > 0);
        if (index_str == "faiss_brute_force")
            return get_nearest_nodes_brute_force(nodes_orig, k);

        std::vector<idx_t> return_nns(k * nodes_orig.size());
        std::vector<float> return_distances(k * nodes_orig.size());

        std::unordered_map<idx_t, idx_t> node_map;
        std::unordered_map<idx_t, u_int32_t> nns_count;
        node_map.reserve(nodes_orig.size());
        nns_count.reserve(nodes_orig.size());
        for (size_t c = 0; c < nodes_orig.size(); ++c)
        {
            node_map.insert({this->get_orig_to_internal_node_mapping(nodes_orig[c]), c});
            nns_count.insert({this->get_orig_to_internal_node_mapping(nodes_orig[c]), 0});
        }

        for(size_t _nr_lookups=k+1; _nr_lookups<2+2*max_id_nr(); _nr_lookups*=2)
        {
            const size_t nr_lookups = std::min(_nr_lookups, size_t(index->ntotal));
            //std::cout << "[feature index get_nearest_nodes] nr lookups = " << nr_lookups << "\n";
            if(node_map.size() > 0)
            {
                std::vector<idx_t> cur_nodes;
                for(const auto [node, idx] : node_map)
                    cur_nodes.push_back(node);
                std::vector<idx_t> nns(cur_nodes.size() * nr_lookups);
                std::vector<float> distances(cur_nodes.size() * nr_lookups);

                std::vector<float> query_features(node_map.size()*d);
                for(size_t c=0; c<cur_nodes.size(); ++c)
                {
                    for(size_t l=0; l<d; ++l)
                        query_features[c*d + l] = features[cur_nodes[c]*d + l];
                    if(track_dist_offset_)
                        query_features[c*d + d-1] *= -1.0;
                }

                {
                    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("faiss search");
                    index->search(cur_nodes.size(), query_features.data(), nr_lookups, distances.data(), nns.data());
                }

                for(size_t c=0; c<cur_nodes.size(); ++c)
                {
                    size_t nns_count = 0;
                    for(size_t l=0; l<nr_lookups; ++l)
                    {
                        if(nns[c*nr_lookups + l] >= 0 && nns[c*nr_lookups + l] < active.size() && nns[c*nr_lookups + l] != cur_nodes[c] && active[nns[nr_lookups*c + l]] == true)
                        {
                            assert(node_map.count(cur_nodes[c]) > 0);
                            return_nns[node_map[cur_nodes[c]] * k + nns_count] = this->get_internal_to_orig_node_mapping(nns[c*nr_lookups + l]);
                            return_distances[node_map[cur_nodes[c]] * k + nns_count] = distances[c*nr_lookups + l];
                            nns_count++;
                            if(nns_count == k)
                            {
                                node_map.erase(cur_nodes[c]);
                                break;
                            }
                        }
                    }
                }
            }
        }

        for(size_t i=0; i<nodes_orig.size(); ++i)
        {
            for(size_t l=0; l<k; ++l)
            {
                assert(return_nns[i*k + l] != nodes_orig[i]);
                assert(return_nns[i*k + l] != -1);
            }
            for(size_t l=0; l+1<k; ++l)
            {
                assert(return_distances[i*k + l] >= return_distances[i*k + l+1]);
            }
        }
        return {return_nns, return_distances};
    }
    
    std::tuple<std::vector<idx_t>, std::vector<float>> feature_index_faiss::get_nearest_nodes_brute_force(const std::vector<idx_t>& nodes, const size_t k) const
    {
        const auto num_query_nodes = nodes.size();
        std::vector<float> query_features(num_query_nodes * d);
        for (size_t c = 0; c != num_query_nodes; ++c)
        {
            const auto internal_node = this->get_orig_to_internal_node_mapping(nodes[c]);
            std::copy(features.begin() + internal_node * d, features.begin() + (internal_node + 1) * d, query_features.begin() + c * d);
            if (track_dist_offset_)
                query_features[c * d + d - 1] *= -1.0;
        }
        std::vector<float> distances(num_query_nodes * (k + 1));
        std::vector<idx_t> ids(num_query_nodes * (k + 1));
        knn_inner_product(query_features.data(), features.data(), d, num_query_nodes, this->max_id_nr() + 1, k + 1, distances.data(), ids.data(), 
                        reinterpret_cast<const faiss::IDSelector*>(id_selector));

        std::vector<float> final_distances(num_query_nodes * k);
        std::vector<idx_t> final_ids(num_query_nodes * k);

        size_t index_1d = 0;
        size_t out_index_1d = 0;
        for (size_t c = 0; c != num_query_nodes; ++c)
        {
            const auto self = nodes[c];
            size_t num_added = 0;
            for (size_t n = 0; n != k + 1; ++n, ++index_1d)
            {
                if (num_added == k)
                    continue;

                const auto neighbour = this->get_internal_to_orig_node_mapping(ids[index_1d]);
                if (neighbour == self)
                    continue;
                // distance ideally should be distances[index_1d] but it is not accurate due to floating point calculation.
                final_distances[out_index_1d] = this->inner_product(nodes[c], neighbour); 
                // assert(std::abs(this->inner_product(self, neighbour) - distances[index_1d]) <= );
                final_ids[out_index_1d] = neighbour;
                out_index_1d++;
                num_added++;
            }
        }
        return {final_ids, final_distances};
    }

    double feature_index_faiss::inner_product(const idx_t i_orig, const idx_t j_orig) const
    {
        const idx_t i = get_orig_to_internal_node_mapping(i_orig);
        const idx_t j = get_orig_to_internal_node_mapping(j_orig);
        assert(i < active.size());
        assert(j < active.size());
        double x = faiss::fvec_inner_product(&features[i*d], &features[j*d], d - 1);
        if(track_dist_offset_)
            x -= features[i*d+d-1]*features[j*d+d-1];
        else
            x += features[i*d+d-1]*features[j*d+d-1];
        return x;
    }

    feature_index_faiss::~feature_index_faiss()
    {
        delete id_selector;
    }
}
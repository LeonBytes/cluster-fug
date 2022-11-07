#include "feature_index.h"
#include "time_measure_util.h"
#include <faiss/index_factory.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <unordered_map>
//#include <iostream>

namespace DENSE_MULTICUT {

    feature_index::feature_index(const size_t _d, const size_t n, const std::vector<float>& _features, const std::string& _index_str, const bool track_dist_offset)
        : d(_d),
        features(_features),
        index(index_factory(d, _index_str.c_str(), faiss::MetricType::METRIC_INNER_PRODUCT)),
        nr_active(n),
        track_dist_offset_(track_dist_offset),
        index_str(_index_str)
    {
        index->train(n, features.data());
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("faiss add");
            index->add(n, features.data());
        }

        active = std::vector<char>(n, true);
        vacant_node = n;
        internal_to_orig_node_mapping = std::vector<faiss::Index::idx_t>(2 * n);
        orig_to_internal_node_mapping = std::vector<faiss::Index::idx_t>(2 * n);
        std::iota(internal_to_orig_node_mapping.begin(), internal_to_orig_node_mapping.end(), 0);
        std::iota(orig_to_internal_node_mapping.begin(), orig_to_internal_node_mapping.end(), 0);
        mapping_is_identity = true;
    }

    std::tuple<faiss::Index::idx_t, float> feature_index::get_nearest_node(const faiss::Index::idx_t id_orig)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
        const size_t id = get_orig_to_internal_node_mapping(id_orig);
        if (track_dist_offset_)
            features[id * d + d - 1] *= -1.0;
        faiss::Index::idx_t nearest_node = -1;
        float nearest_distance = 0.0; 
        for (size_t nr_lookups = 2; nr_lookups < 2 * index->ntotal && nearest_node == -1; nr_lookups *= 2)
        {
            float distance[std::min(nr_lookups, size_t(index->ntotal))];
            faiss::Index::idx_t nns[std::min(nr_lookups, size_t(index->ntotal))];
            index->search(1, features.data() + id * d, std::min(nr_lookups, size_t(index->ntotal)), distance, nns);
            assert(std::is_sorted(distance, distance + std::min(nr_lookups, size_t(index->ntotal)), std::greater<float>()));
            for (size_t k = 0; k < std::min(nr_lookups, size_t(index->ntotal)); ++k)
                if (nns[k] < active.size() && nns[k] != id && active[nns[k]] == true)
                {
                    nearest_node = get_internal_to_orig_node_mapping(nns[k]);
                    nearest_distance = distance[k];
                    break;
                }
        }
        if (nearest_node == -1)
            throw std::runtime_error("Could not find nearest neighbor");

        if (track_dist_offset_)
            features[id * d + d - 1] *= -1.0;

        return {nearest_node, nearest_distance};
    }

    std::tuple<std::vector<faiss::Index::idx_t>, std::vector<float>> feature_index::get_nearest_nodes(const std::vector<faiss::Index::idx_t> &nodes_orig) const
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("faiss get nearest nodes");
        assert(nodes_orig.size() > 0);
        //std::cout << "[feature index] search nearest neighbors for " << nodes.size() << " nodes\n";

        std::vector<faiss::Index::idx_t> return_nns(nodes_orig.size());
        std::vector<float> return_distances(nodes_orig.size());

        std::unordered_map<faiss::Index::idx_t, faiss::Index::idx_t> node_map;
        node_map.reserve(nodes_orig.size());
        for (size_t c = 0; c < nodes_orig.size(); ++c)
            node_map.insert({get_orig_to_internal_node_mapping(nodes_orig[c]), c});

        for (size_t _nr_lookups = 2; _nr_lookups < 2 + 2 * max_id_nr(); _nr_lookups *= 2)
        {
            const size_t nr_lookups = std::min(_nr_lookups, size_t(index->ntotal));
            if (node_map.size() > 0)
            {
                std::vector<faiss::Index::idx_t> cur_nodes;
                for (const auto [node, idx] : node_map)
                    cur_nodes.push_back(node);
                std::vector<faiss::Index::idx_t> nns(cur_nodes.size() * nr_lookups);
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
                            return_nns[node_map[cur_nodes[c]]] = get_internal_to_orig_node_mapping(nns[c * nr_lookups + k]);
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

    std::tuple<std::vector<faiss::Index::idx_t>, std::vector<float>> feature_index::get_nearest_nodes(const std::vector<faiss::Index::idx_t>& nodes_orig, const size_t k) const
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("faiss get k nearest nodes");
        assert(k > 0);
        assert(k < nr_nodes());
        assert(nodes_orig.size() > 0);

        std::vector<faiss::Index::idx_t> return_nns(k * nodes_orig.size());
        std::vector<float> return_distances(k * nodes_orig.size());

        std::unordered_map<faiss::Index::idx_t, faiss::Index::idx_t> node_map;
        std::unordered_map<faiss::Index::idx_t, u_int32_t> nns_count;
        node_map.reserve(nodes_orig.size());
        nns_count.reserve(nodes_orig.size());
        for (size_t c = 0; c < nodes_orig.size(); ++c)
        {
            node_map.insert({get_orig_to_internal_node_mapping(nodes_orig[c]), c});
            nns_count.insert({get_orig_to_internal_node_mapping(nodes_orig[c]), 0});
        }

        for(size_t _nr_lookups=k+1; _nr_lookups<2+2*max_id_nr(); _nr_lookups*=2)
        {
            const size_t nr_lookups = std::min(_nr_lookups, size_t(index->ntotal));
            //std::cout << "[feature index get_nearest_nodes] nr lookups = " << nr_lookups << "\n";
            if(node_map.size() > 0)
            {
                std::vector<faiss::Index::idx_t> cur_nodes;
                for(const auto [node, idx] : node_map)
                    cur_nodes.push_back(node);
                std::vector<faiss::Index::idx_t> nns(cur_nodes.size() * nr_lookups);
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
                            return_nns[node_map[cur_nodes[c]] * k + nns_count] = get_internal_to_orig_node_mapping(nns[c*nr_lookups + l]);
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

    void feature_index::remove(const faiss::Index::idx_t i_orig)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
        const faiss::Index::idx_t i = get_orig_to_internal_node_mapping(i_orig);
        assert(i < active.size());
        assert(active[i] == true);
        active[i] = false;
        nr_active--;
    }

    faiss::Index::idx_t feature_index::merge(const faiss::Index::idx_t i_orig, const faiss::Index::idx_t j_orig, const bool add_to_index)
    {
        MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME;
        const faiss::Index::idx_t i = get_orig_to_internal_node_mapping(i_orig);
        const faiss::Index::idx_t j = get_orig_to_internal_node_mapping(j_orig);
        assert(i != j);
        assert(i < active.size());
        assert(j < active.size());

        active[i] = false;
        active[j] = false;

        nr_active--;

        const faiss::Index::idx_t new_id = features.size()/d;
        for(size_t l=0; l<d; ++l)
            features.push_back(features[i*d + l] + features[j*d + l]);
        if (add_to_index)
            index->add(1, features.data() + new_id*d);
        active.push_back(true);
        mapping_is_identity = false;
        internal_to_orig_node_mapping[new_id] = vacant_node;
        orig_to_internal_node_mapping[vacant_node] = new_id;
        return vacant_node++;
    }

    double feature_index::inner_product(const faiss::Index::idx_t i_orig, const faiss::Index::idx_t j_orig) const
    {
        const faiss::Index::idx_t i = get_orig_to_internal_node_mapping(i_orig);
        const faiss::Index::idx_t j = get_orig_to_internal_node_mapping(j_orig);
        assert(i < active.size());
        assert(j < active.size());
        float x = 0.0;
        for(size_t l=0; l<d-1; ++l)
            x += features[i*d+l]*features[j*d+l];
        if(track_dist_offset_)
            x -= features[i*d+d-1]*features[j*d+d-1];
        else
            x += features[i*d+d-1]*features[j*d+d-1];
        return x;
    }

    bool feature_index::node_active(const faiss::Index::idx_t i_orig) const
    {
        const faiss::Index::idx_t idx = get_orig_to_internal_node_mapping(i_orig);
        assert(idx < active.size());
        return active[idx] == true;
    }

    size_t feature_index::max_id_nr() const 
    { 
        assert(active.size() > 0);
        return active.size()-1;
    }

    size_t feature_index::nr_nodes() const
    {
        assert(nr_active == std::count(active.begin(), active.end(), true));
        return nr_active;
    }

    std::vector<faiss::Index::idx_t> feature_index::get_active_nodes() const
    {
        std::vector<faiss::Index::idx_t> active_nodes;
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

    faiss::Index::idx_t feature_index::get_orig_to_internal_node_mapping(const faiss::Index::idx_t i) const
    {
        if (mapping_is_identity)
            return i;
        return orig_to_internal_node_mapping[i];
    }

    faiss::Index::idx_t feature_index::get_internal_to_orig_node_mapping(const faiss::Index::idx_t i) const
    {
        if (mapping_is_identity)
            return i;
        return internal_to_orig_node_mapping[i];
    }

    // std::tuple<std::vector<float>, std::vector<faiss::Index::idx_t>> feature_index::reconstruct_clean_index(const std::vector<faiss::Index::idx_t>& orig_node_ids) const
    // {
    //     std::vector<faiss::Index::idx_t> new_node_ids(orig_node_ids);
    //     std::vector<float> active_features(nr_active * d);
    //     int i_new = 0;
    //     for (int i = 0; i != active.size(); ++i)
    //     {
    //         if (active[i])
    //         {
    //             new_node_ids[i_new] = orig_node_ids[i];
    //             std::copy(features.begin() + i * d, features.begin() + (i + 1) * d, active_features.begin() + (i_new * d));
    //             ++i_new;
    //         }
    //     }
    //     return {active_features, new_node_ids};
    // }

    void feature_index::reconstruct_clean_index()
    {
        std::vector<float> active_features(nr_active * d);

        int i_internal_new = 0;
        for (int i = 0; i != active.size(); ++i)
        {
            if (active[i])
            {
                const auto i_orig = get_internal_to_orig_node_mapping(i);
                orig_to_internal_node_mapping[i_orig] = i_internal_new;
                internal_to_orig_node_mapping[i_internal_new] = i_orig;
                std::copy(features.begin() + i * d, features.begin() + (i + 1) * d, active_features.begin() + (i_internal_new * d));
                active[i] = false;
                ++i_internal_new;
            }
        }
        std::swap(features, active_features);
        index.reset(index_factory(d, index_str.c_str(), faiss::MetricType::METRIC_INNER_PRODUCT));
        index->train(nr_active, features.data());
        active.resize(nr_active);
        std::fill(active.begin(), active.end(), true);        
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("faiss add");
            index->add(nr_active, features.data());
        }

        mapping_is_identity = false;
    }
}
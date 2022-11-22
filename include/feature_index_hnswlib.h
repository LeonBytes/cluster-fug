#pragma once
#include "feature_index.h"
#include <hnswlib/hnswlib.h>

namespace DENSE_MULTICUT {

    class feature_index_hnswlib: public feature_index<float> {
        public:
            feature_index_hnswlib(const size_t d, const size_t n, const std::vector<float>& _features, const std::string& index_str, const bool track_dist_offset = false);

            virtual std::tuple<idx_t, float> get_nearest_node(const idx_t node) const;
            virtual std::tuple<std::vector<idx_t>, std::vector<float>> get_nearest_nodes(const std::vector<idx_t>& nodes) const;
            virtual std::tuple<std::vector<idx_t>, std::vector<float>> get_nearest_nodes(const std::vector<idx_t>& nodes, const size_t k) const;
            // virtual std::tuple<idx_t, float> get_nearest_node(const idx_t node) const;
            // virtual idx_t merge(const idx_t i, const idx_t j, const bool add_to_index = true);
            // virtual void reconstruct_clean_index();

            virtual ~feature_index_hnswlib();


        private:
            std::tuple<std::vector<idx_t>, std::vector<float>> get_nearest_nodes_brute_force(const std::vector<idx_t>& nodes, const size_t k) const;
            const std::string index_str;
            hnswlib::AlgorithmInterface<float>* index;
    };
}

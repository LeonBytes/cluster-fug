#pragma once
#include <faiss/Index.h>
#include <vector>
#include <tuple>
#include <memory>

namespace DENSE_MULTICUT {

    using idx_t = faiss::Index::idx_t;

    template<typename REAL>
    class feature_index {
        public:
            feature_index() {}
            feature_index(const size_t d, const size_t n, const std::vector<REAL>& _features, const bool track_dist_offset = false);

            virtual std::tuple<std::vector<idx_t>, std::vector<REAL>> get_nearest_nodes(const std::vector<idx_t>& nodes) const = 0;
            virtual std::tuple<std::vector<idx_t>, std::vector<REAL>> get_nearest_nodes(const std::vector<idx_t>& nodes, const size_t k) const = 0;
            virtual std::tuple<idx_t, REAL> get_nearest_node(const idx_t node) const = 0;

            virtual void reconstruct_clean_index(std::string new_index_str="");
            virtual idx_t merge(const idx_t i, const idx_t j, const bool add_to_index = true);
            virtual void remove(const idx_t i);

            double inner_product(const idx_t i, const idx_t j) const;
            bool node_active(const idx_t idx) const;
            size_t max_id_nr() const;
            size_t nr_nodes() const;
            std::vector<idx_t> get_active_nodes() const;

            idx_t get_orig_to_internal_node_mapping(const idx_t i) const;
            idx_t get_internal_to_orig_node_mapping(const idx_t i) const;

            virtual ~feature_index() {}

        protected:
            size_t d;
            std::vector<REAL> features;
            std::vector<char> active;
            std::vector<idx_t> internal_to_orig_node_mapping;
            std::vector<idx_t> orig_to_internal_node_mapping;
            bool mapping_is_identity = true;
            size_t nr_active = 0;
            idx_t vacant_node = -1;
            bool track_dist_offset_ = false;
    };
}

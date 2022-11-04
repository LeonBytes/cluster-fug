#pragma once
#include <faiss/Index.h>
#include <vector>
#include <tuple>
#include <memory>

namespace DENSE_MULTICUT {

    class feature_index {
        public:
            feature_index(const size_t d, const size_t n, const std::vector<float>& _features, const std::string& index_str, const bool track_dist_offset = false);

            void remove(const faiss::Index::idx_t i);
            faiss::Index::idx_t merge(const faiss::Index::idx_t i, const faiss::Index::idx_t j);
            double inner_product(const faiss::Index::idx_t i, const faiss::Index::idx_t j) const;
            std::tuple<std::vector<faiss::Index::idx_t>, std::vector<float>> get_nearest_nodes(const std::vector<faiss::Index::idx_t>& nodes) const;
            std::tuple<std::vector<faiss::Index::idx_t>, std::vector<float>> get_nearest_nodes(const std::vector<faiss::Index::idx_t>& nodes, const size_t k) const;
            std::tuple<faiss::Index::idx_t, float> get_nearest_node(const faiss::Index::idx_t node);

            bool node_active(const faiss::Index::idx_t idx) const;
            size_t max_id_nr() const;
            size_t nr_nodes() const;
            std::vector<faiss::Index::idx_t> get_active_nodes() const;
            std::tuple<std::vector<float>, std::vector<faiss::Index::idx_t>> reconstruct_clean_index(const std::vector<faiss::Index::idx_t>& orig_node_ids) const;
        private:
            size_t d;
            std::unique_ptr<faiss::Index> index;
            std::vector<float> features;
            std::vector<char> active;
            size_t nr_active = 0;
            bool can_remove = false;
            bool track_dist_offset_ = false;
    };
}

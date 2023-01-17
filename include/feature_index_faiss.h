#pragma once
#include <faiss/Index.h>
#include <faiss/impl/IDSelector.h>
#include "feature_index.h"

namespace DENSE_MULTICUT {

    using idx_t = faiss::Index::idx_t;
    struct IDSelectorCustom : faiss::IDSelector {
        size_t n;
        std::vector<char> active;

        IDSelectorCustom() {}
        IDSelectorCustom(size_t n);
        bool is_member(idx_t id) const final;
        void merge(const idx_t i, const idx_t j);
        ~IDSelectorCustom() override {}
    };


    class feature_index_faiss: public feature_index<float> {
        public:
            feature_index_faiss(const size_t d, const size_t n, const std::vector<float>& _features, const std::string& index_str, const bool track_dist_offset = false);

            virtual std::tuple<std::vector<idx_t>, std::vector<float>> get_nearest_nodes(const std::vector<idx_t>& nodes) const;
            virtual std::tuple<std::vector<idx_t>, std::vector<float>> get_nearest_nodes(const std::vector<idx_t>& nodes, const size_t k) const;
            virtual std::tuple<idx_t, float> get_nearest_node(const idx_t node) const;
            virtual idx_t merge(const idx_t i, const idx_t j, const bool add_to_index = true);
            virtual void reconstruct_clean_index(std::string new_index_str = "");

            double inner_product(const idx_t i, const idx_t j) const;

            virtual ~feature_index_faiss();


        private:
            std::tuple<std::vector<idx_t>, std::vector<float>> get_nearest_nodes_brute_force(const std::vector<idx_t>& nodes, const size_t k) const;

            std::shared_ptr<faiss::Index> index;
            std::string index_str;
            IDSelectorCustom* id_selector;
    };
}

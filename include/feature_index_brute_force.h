#pragma once
#include <faiss/Index.h>
#include "feature_index.h"

namespace DENSE_MULTICUT {

    template <typename REAL>
    class feature_index_brute_force: public feature_index<REAL> {
        public:
            using feature_index<REAL>::feature_index; // inherit base constructor

            virtual std::tuple<idx_t, REAL> get_nearest_node(const idx_t node) const;
            virtual std::tuple<std::vector<idx_t>, std::vector<REAL>> get_nearest_nodes(const std::vector<idx_t>& nodes) const;
            virtual std::tuple<std::vector<idx_t>, std::vector<REAL>> get_nearest_nodes(const std::vector<idx_t>& nodes, const size_t k) const;

            virtual ~feature_index_brute_force() {}

    };
}

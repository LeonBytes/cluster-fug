#include <vector>
#include <cstddef>
#include <string>

namespace DENSE_MULTICUT {

    std::vector<size_t> dense_gaec_inc_nn_faiss(const size_t n, const size_t d, const std::vector<float>& features, const std::string index_str, const bool track_dist_offset, const size_t k_in, const size_t k_cap);

    template<typename REAL>
    std::vector<size_t> dense_gaec_inc_nn_brute_force(const size_t n, const size_t d, const std::vector<REAL>& features, const bool track_dist_offset, const size_t k_in, const size_t k_cap);

}

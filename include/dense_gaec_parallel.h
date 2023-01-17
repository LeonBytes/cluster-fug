#include <vector>
#include <cstddef>

namespace DENSE_MULTICUT {

    std::vector<size_t> dense_gaec_parallel_faiss(const size_t n, const size_t d, std::vector<float> features, const std::string index_str, const bool track_dist_offset = false);

    template<typename REAL>
    std::vector<size_t> dense_gaec_parallel_brute_force(const size_t n, const size_t d, std::vector<REAL> features, const bool track_dist_offset = false);

}


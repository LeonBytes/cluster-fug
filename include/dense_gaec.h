#include <vector>
#include <cstddef>
#include <string>

namespace DENSE_MULTICUT {

    std::vector<size_t> dense_gaec_faiss(const size_t n, const size_t d, const std::vector<float>& features, const std::string index_str, const bool track_dist_offset);

    template<typename REAL>
    std::vector<size_t> dense_gaec_brute_force(const size_t n, const size_t d, const std::vector<REAL>& features, const bool track_dist_offset = false);

}

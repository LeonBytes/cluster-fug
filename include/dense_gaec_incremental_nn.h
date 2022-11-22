#include <vector>
#include <cstddef>
#include <string>
namespace DENSE_MULTICUT {

    std::vector<size_t> dense_gaec_incremental_nn_faiss(const size_t n, const size_t d, std::vector<float> features, const std::string index_str, const bool track_dist_offset, const size_t k_in);

    std::vector<size_t> dense_gaec_incremental_nn_hnswlib(const size_t n, const size_t d, std::vector<float> features, const std::string index_str, const bool track_dist_offset, const size_t k_in);

    template<typename REAL>
    std::vector<size_t> dense_gaec_incremental_nn_brute_force(const size_t n, const size_t d, std::vector<REAL> features, const bool track_dist_offset, const size_t k_in);
}

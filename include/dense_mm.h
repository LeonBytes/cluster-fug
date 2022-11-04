#include <vector>
#include <cstddef>

namespace DENSE_MULTICUT {

    std::vector<size_t> dense_mm_flat_index(const size_t n, const size_t d, std::vector<float> features, const bool track_dist_offset = false);

    std::vector<size_t> dense_mm_hnsw(const size_t n, const size_t d, std::vector<float> features, const bool track_dist_offset = false);

}

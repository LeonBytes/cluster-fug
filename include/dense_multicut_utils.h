#include <vector>
#include <cstddef>

namespace DENSE_MULTICUT {

    double cost_disconnected(const size_t n, const size_t d, const std::vector<float>& features, const bool track_dist_offset = false);
    std::vector<float> append_dist_offset_in_features(const std::vector<float>& features, const float dist_offset, const size_t n, const size_t d);
    double labeling_cost(const std::vector<size_t>& labeling, const size_t n, const size_t d, const std::vector<float>& features, const bool track_dist_offset = false);

}

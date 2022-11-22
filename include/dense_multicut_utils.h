#include <vector>
#include <cstddef>

namespace DENSE_MULTICUT {

    template<typename REAL>
    double cost_disconnected(const size_t n, const size_t d, const std::vector<REAL>& features, const bool track_dist_offset = false);
    
    template<typename REAL>
    std::vector<REAL> append_dist_offset_in_features(const std::vector<REAL>& features, const double dist_offset, const size_t n, const size_t d);
    
    template<typename REAL>
    double labeling_cost(const std::vector<size_t>& labeling, const size_t n, const size_t d, const std::vector<REAL>& features, const bool track_dist_offset = false);

}

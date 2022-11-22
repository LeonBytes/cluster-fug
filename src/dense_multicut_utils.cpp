#include "dense_multicut_utils.h"
#include <iostream>
#include <cmath>

namespace DENSE_MULTICUT {

    template<typename REAL>
    double cost_disconnected(const size_t n, const size_t d, const std::vector<REAL>& features, const bool track_dist_offset)
    {
        const size_t d_eff = track_dist_offset ? d - 1: d;
        std::vector<double> feature_sum(d_eff);
        for(size_t i=0; i<n; ++i)
            for(size_t l=0; l<d_eff; ++l)
                feature_sum[l] += features[i*d+l];

        double cost = 0.0;

        for(size_t l=0; l<d_eff; ++l)
            cost += feature_sum[l] * feature_sum[l];

        // remove diagonal entries (self-edge)
        for(size_t i=0; i<n; ++i)
            for(size_t l=0; l<d_eff; ++l)
                cost -= features[i*d+l]*features[i*d+l];

        cost /= 2.0;
        // account for offset term:
        if (track_dist_offset)
        {
            const float dist_offset_sqrt = features[d - 1];
            cost -= dist_offset_sqrt * dist_offset_sqrt * n * (n - 1) / 2.0;
        }
        std::cout << "disconnected multicut cost = " << cost << "\n";
        return cost;
    }

    template<typename REAL>
    std::vector<REAL> append_dist_offset_in_features(const std::vector<REAL>& features, const double dist_offset, const size_t n, const size_t d)
    {
        std::vector<REAL> features_w_dist_offset(n * (d + 1));
        if (dist_offset < 0)
            throw std::runtime_error("dist_offset can only be >= 0.");
        std::cout << "Accounting for dist_offset = " << dist_offset << " by adding additional feature dimension.\n";
        for(size_t i=0; i<n; ++i)
        {
            for(size_t l=0; l<d; ++l)
                features_w_dist_offset[i * (d + 1) + l] = features[i * d + l];
            features_w_dist_offset[i * (d + 1) + d] = std::sqrt(dist_offset);
        }
        return features_w_dist_offset;
    }

    template<typename REAL>
    double labeling_cost(const std::vector<size_t>& labeling, const size_t n, const size_t d, const std::vector<REAL>& features, const bool track_dist_offset)
    {
        double cost = 0.0;
        const size_t d_eff = track_dist_offset ? d - 1: d;
        const double offset = features[d - 1] * features[d - 1];
        for(size_t i=0; i<n; ++i)
        {
            const size_t li = labeling[i];
            for(size_t j=i+1; j<n; ++j)
            {
                const size_t lj = labeling[j];
                if (li == lj) continue;
                for(size_t l=0; l<d_eff; ++l)
                    cost += features[i * d + l] * features[j * d + l];
                if (track_dist_offset)
                    cost -= offset;
            }
        }
        return cost;
    }

    template double cost_disconnected<float>(const size_t n, const size_t d, const std::vector<float>& features, const bool track_dist_offset);
    template double cost_disconnected<double>(const size_t n, const size_t d, const std::vector<double>& features, const bool track_dist_offset);

    template std::vector<float> append_dist_offset_in_features<float>(const std::vector<float>& features, const double dist_offset, const size_t n, const size_t d);
    template std::vector<double> append_dist_offset_in_features<double>(const std::vector<double>& features, const double dist_offset, const size_t n, const size_t d);

    template double labeling_cost<float>(const std::vector<size_t>& labeling, const size_t n, const size_t d, const std::vector<float>& features, const bool track_dist_offset);
    template double labeling_cost<double>(const std::vector<size_t>& labeling, const size_t n, const size_t d, const std::vector<double>& features, const bool track_dist_offset);

}

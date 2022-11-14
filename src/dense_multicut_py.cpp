#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dense_gaec.h"
#include "dense_gaec_parallel.h"
#include "dense_gaec_adj_matrix.h"
#include "dense_gaec_incremental_nn.h"
#include "dense_features_parser.h"
#include "dense_multicut_utils.h"

namespace py=pybind11;
using namespace DENSE_MULTICUT;

PYBIND11_MODULE(dense_multicut_py, m) {
    m.doc() = "Bindings for dense multicut. Available solver types: \n "
    "adj_matrix, flat_index, hnsw, mm_flat_index, mm_hnsw, parallel_flat_index, parallel_hnsw,\n"
    "flat_index, inc_nn_flat, inc_nn_hnsw. \n";
    m.def("dense_gaec_incremental_nn", [](std::vector<float> features, const int num_nodes, int dim, const float dist_offset, const size_t k_inc_nn, const std::string solver_type) 
    {
        bool track_dist_offset = false;
        if (dist_offset != 0.0)
        {
            std::cout << "[dense multicut] use distance offset\n";
            features = append_dist_offset_in_features(features, dist_offset, num_nodes, dim);
            dim += 1;
            track_dist_offset = true;
        }

        std::vector<size_t> labeling;
        if (solver_type ==  "adj_matrix")
            labeling = dense_gaec_adj_matrix(num_nodes, dim, features, track_dist_offset);
        else if (solver_type ==  "flat_index")
            labeling = dense_gaec_flat_index(num_nodes, dim, features, track_dist_offset);
        else if (solver_type ==  "hnsw")
            labeling = dense_gaec_hnsw(num_nodes, dim, features, track_dist_offset);
        else if (solver_type ==  "parallel_flat_index")
            labeling = dense_gaec_parallel_flat_index(num_nodes, dim, features, track_dist_offset);
        else if (solver_type ==  "parallel_hnsw")
            labeling = dense_gaec_parallel_hnsw(num_nodes, dim, features, track_dist_offset);
        else if (solver_type ==  "flat_index")
            labeling = dense_gaec_flat_index(num_nodes, dim, features, track_dist_offset);
        else if (solver_type ==  "inc_nn_flat")
            labeling = dense_gaec_incremental_nn(num_nodes, dim, features, k_inc_nn, "Flat", track_dist_offset);
        else if (solver_type ==  "inc_nn_hnsw")
            labeling = dense_gaec_incremental_nn(num_nodes, dim, features, k_inc_nn, "HNSW64", track_dist_offset);
        else
            throw std::runtime_error("Unknown solver type: " + solver_type);

        return labeling;
    });
}
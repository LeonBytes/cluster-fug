#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dense_gaec.h"
#include "dense_gaec_inc_nn.h"
#include "dense_gaec_parallel.h"
#include "dense_gaec_adj_matrix.h"
#include "dense_laec_inc_nn.h"
#include "dense_features_parser.h"
#include "dense_multicut_utils.h"

namespace py=pybind11;
using namespace DENSE_MULTICUT;

std::vector<size_t> run_faiss_solvers(const std::vector<float> features, const size_t num_nodes, const size_t dim,
                                     const std::string solver_type, const std::string index_str, const int k_inc_nn = 10, 
                                     const bool track_dist_offset = false, const size_t k_cap = 10)
{
    std::vector<size_t> labeling;
    if (solver_type ==  "gaec")
        labeling = dense_gaec_faiss(num_nodes, dim, features, index_str, track_dist_offset);
    else if (solver_type ==  "gaec_inc")
        labeling = dense_gaec_inc_nn_faiss(num_nodes, dim, features, index_str, track_dist_offset, k_inc_nn, k_cap);
    else if (solver_type ==  "parallel")
        labeling = dense_gaec_parallel_faiss(num_nodes, dim, features, index_str, track_dist_offset);
    else if (solver_type ==  "laec")
        labeling = dense_laec_inc_nn_faiss(num_nodes, dim, features, index_str, track_dist_offset, k_inc_nn, k_cap);
    else if (solver_type ==  "laec_bf_later")
        labeling = dense_laec_inc_nn_faiss_bf_after(num_nodes, dim, features, index_str, track_dist_offset, k_inc_nn, k_cap);
    else
        throw std::runtime_error("Unknown solver type: " + solver_type);
    return labeling;
}

template <typename REAL>
std::vector<size_t> run_brute_force_solvers(const std::vector<REAL> features, const size_t num_nodes, const size_t dim, 
                    const std::string solver_type, const int k_inc_nn = 10, const bool track_dist_offset = false, const size_t k_cap = 10)
{
    std::vector<size_t> labeling;
    if (solver_type ==  "adj_matrix")
        labeling = dense_gaec_adj_matrix<REAL>(num_nodes, dim, features, track_dist_offset);
    else if (solver_type ==  "gaec")
        labeling = dense_gaec_brute_force<REAL>(num_nodes, dim, features, track_dist_offset);
    else if (solver_type ==  "gaec_inc")
        labeling = dense_gaec_inc_nn_brute_force<REAL>(num_nodes, dim, features, track_dist_offset, k_inc_nn, k_cap);
    else if (solver_type ==  "parallel")
        labeling = dense_gaec_parallel_brute_force<REAL>(num_nodes, dim, features, track_dist_offset);
    else if (solver_type ==  "laec")
        labeling = dense_laec_inc_nn_brute_force<REAL>(num_nodes, dim, features, track_dist_offset, k_inc_nn, k_cap);
    else
        throw std::runtime_error("Unknown solver type: " + solver_type);
    return labeling;
}

PYBIND11_MODULE(dense_multicut_py, m) {
    m.doc() = "Bindings for dense multicut. Available index types: \n"
    "brute_force\n, faiss_brute_force\n, Flat\n, HNSW\n"
    "Available solver types: \n "
    "adj_matrix\n, gaec\n, gaec_inc\n, parallel\n, laec, laec_bf_later\n"
    "all index except brute_force need float, brute_force also suppports double.";
    m.def("dense_multicut_double", [](
        std::vector<double> features, const int num_nodes, int dim, const float dist_offset, const size_t k_inc_nn, const std::string index_type, const std::string solver_type, const size_t k_cap) 
    {
        if (index_type != "brute_force")
            throw std::runtime_error("Unknown index_type: " + index_type + " (only brute_force is supported)");
        std::cout<<"Using double precision\n";
        bool track_dist_offset = false;
        if (dist_offset != 0.0)
        {
            features = append_dist_offset_in_features<double>(features, dist_offset, num_nodes, dim);
            dim += 1;
            track_dist_offset = true;
        }
        std::vector<size_t> labeling = run_brute_force_solvers<double>(features, num_nodes, dim, solver_type, k_inc_nn, dist_offset, k_cap);
        return labeling;
    });
    m.def("dense_multicut", [](
        std::vector<float> features, const int num_nodes, int dim, const float dist_offset, const size_t k_inc_nn, const std::string index_type, const std::string solver_type, const size_t k_cap) 
    {
        std::cout<<"Using float precision\n";
        bool track_dist_offset = false;
        std::vector<size_t> labeling;
        if (dist_offset != 0.0)
        {
            features = append_dist_offset_in_features<float>(features, dist_offset, num_nodes, dim);
            dim += 1;
            track_dist_offset = true;
        }
        if (index_type == "brute_force")
            labeling = run_brute_force_solvers<float>(features, num_nodes, dim, solver_type, k_inc_nn, dist_offset, k_cap);
        else 
            labeling = run_faiss_solvers(features, num_nodes, dim, solver_type, index_type, k_inc_nn, track_dist_offset, k_cap);

        return labeling;
    });
}
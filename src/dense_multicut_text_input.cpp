#include "dense_gaec.h"
#include "dense_gaec_inc_nn.h"
#include "dense_gaec_parallel.h"
#include "dense_gaec_adj_matrix.h"
#include "dense_laec_inc_nn.h"
#include "dense_features_parser.h"
#include "dense_multicut_utils.h"
#include <iostream>
#include <functional>
#include <CLI/CLI.hpp>

using namespace DENSE_MULTICUT;

template<typename REAL>
std::tuple<std::vector<REAL>, size_t, size_t, bool> get_instance(const std::string file_path, const float dist_offset)
{
    size_t num_nodes, dim;
    std::vector<REAL> features;
    bool track_dist_offset = false;

    std::tie(features, num_nodes, dim) = read_file<REAL>(file_path);
    if (dist_offset != 0.0)
    {
        std::cout << "[dense multicut] use distance offset\n";
        features = append_dist_offset_in_features<REAL>(features, dist_offset, num_nodes, dim);
        dim += 1;
        track_dist_offset = true;
    }
    return {features, num_nodes, dim, track_dist_offset};
}

std::vector<size_t> run_faiss_solvers(const std::vector<float> features, const size_t num_nodes, const size_t dim,
                                     const std::string solver_type, const std::string index_str, const int k_inc_nn = 10, 
                                     const bool track_dist_offset = false, const size_t k_cap = 10)
{
    std::vector<size_t> labeling;
    std::cout<<solver_type<<"\n";
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
    // std::cout<<"Computed cost: "<<labeling_cost<float>(labeling, num_nodes, dim, features, track_dist_offset)<<"\n";
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
    // std::cout<<"Computed cost: "<<labeling_cost<REAL>(labeling, num_nodes, dim, features, track_dist_offset)<<"\n";
    return labeling;
}

int main(int argc, char** argv)
{
    CLI::App app("Dense multicut solvers");

    std::string file_path, solver_type, index_type;
    std::string out_path = "";
    int k_inc_nn = 10, k_cap = 10;
    float dist_offset = 0.0;
    bool use_double = false;
    app.add_option("-f,--file,file_pos", file_path, "Path to dense multicut instance (.txt)")->required()->check(CLI::ExistingPath);
    app.add_option("-i,--index_str,index_pos", index_type, "One of the following feature indexing types: \n"
        "brute_force\n, faiss_brute_force\n, Flat\n, HNSW\n")->required();
    app.add_option("-s,--solver_type,solver_pos", solver_type, "One of the following solvers: \n"
        "adj_matrix\n, gaec\n, gaec_inc, parallel\n, laec\n, laec_bf_later\n")->required();
    app.add_option("-k,--knn,knn_pos", k_inc_nn, "Number of nearest neighbours to build kNN graph. Only used if solver type is incremental (gaec_inc, laec, laec_bf_later)")->check(CLI::PositiveNumber);
    app.add_option("-K,--knn_cap,knn_cap_pos", k_cap, "Number of nearest neighbours to keep in the kNN graph. Only used if solver type is incremental (gaec_inc, laec, laec_bf_later)")->check(CLI::PositiveNumber);
    app.add_option("-t,--thresh,thresh_pos", dist_offset, "Offset to subtract from edge costs, larger value will create more clusters and viceversa.")->check(CLI::NonNegativeNumber);
    app.add_option("-o,--output_file,output_pos", out_path, "Output file path.");
    app.add_flag("-d", use_double, "Use double precision.");

    app.parse(argc, argv);
    std::vector<size_t> labeling;
    if (index_type == "brute_force")
    {
        if (use_double)
        {
            std::cout<<"Using double precision\n";
            const auto [features, num_nodes, dim, track_dist_offset] = get_instance<double>(file_path, dist_offset);
            labeling = run_brute_force_solvers<double>(features, num_nodes, dim, solver_type, k_inc_nn, dist_offset, k_cap);
        }
        else
        {
            const auto [features, num_nodes, dim, track_dist_offset] = get_instance<float>(file_path, dist_offset);
            labeling = run_brute_force_solvers<float>(features, num_nodes, dim, solver_type, k_inc_nn, dist_offset, k_cap);
        }
    }
    else 
    {
        const auto [features, num_nodes, dim, track_dist_offset] = get_instance<float>(file_path, dist_offset);
        labeling = run_faiss_solvers(features, num_nodes, dim, solver_type, index_type, k_inc_nn, track_dist_offset, k_cap);
    }

    if (out_path != "")
    {
        std::ofstream sol_file;
        sol_file.open(out_path);
        std::cout<<"Writing solution to file: "<<out_path<<"\n";
        std::copy(labeling.begin(), labeling.end(), std::ostream_iterator<int>(sol_file, "\n"));
        sol_file.close();
    }   

}
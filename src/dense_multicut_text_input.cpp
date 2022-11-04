#include "dense_gaec.h"
#include "dense_mm.h"
#include "dense_gaec_parallel.h"
#include "dense_gaec_adj_matrix.h"
#include "dense_gaec_incremental_nn.h"
#include "dense_features_parser.h"
#include "dense_multicut_utils.h"
#include <iostream>
#include <functional>
#include <CLI/CLI.hpp>

using namespace DENSE_MULTICUT;

int main(int argc, char** argv)
{
    CLI::App app("Dense multicut solvers");
    std::vector<std::string> available_solvers{"adj_matrix","flat_index", "hnsw", "parallel_flat_index", "parallel_hnsw"};

    std::string file_path, solver_type;
    std::string out_path = "";
    int k_inc_nn = 10;
    float dist_offset = 0.0;
    app.add_option("-f,--file,file_pos", file_path, "Path to dense multicut instance (.txt)")->required()->check(CLI::ExistingPath);
    app.add_option("-s,--solver,solver_pos", solver_type, "One of the following solver types: \n"
        "adj_matrix\n, flat_index\n, hnsw\n, parallel_flat_index\n, parallel_hnsw\n, inc_nn_flat\n, inc_nn_hnsw\n")->required();
    app.add_option("-k,--knn,knn_pos", k_inc_nn, "Number of nearest neighbours to build kNN graph. Only used if solver type is inc_nn")->check(CLI::PositiveNumber);
    app.add_option("-t,--thresh,thresh_pos", dist_offset, "Offset to subtract from edge costs, larger value will create more clusters and viceversa.")->check(CLI::NonNegativeNumber);
    app.add_option("-o,--output_file,output_pos", out_path, "Output file path.");

    app.parse(argc, argv);
    size_t num_nodes, dim;
    std::vector<float> features;
    bool track_dist_offset = false;

    std::tie(features, num_nodes, dim) = read_file(file_path);
    if (dist_offset != 0.0)
    {
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
    else if (solver_type ==  "mm_flat_index")
        labeling = dense_mm_flat_index(num_nodes, dim, features, track_dist_offset);
    else if (solver_type ==  "mm_hnsw")
        labeling = dense_mm_hnsw(num_nodes, dim, features, track_dist_offset);
    else if (solver_type ==  "parallel_flat_index")
        labeling = dense_gaec_parallel_flat_index(num_nodes, dim, features, track_dist_offset);
    else if (solver_type ==  "parallel_hnsw")
        labeling = dense_gaec_parallel_hnsw(num_nodes, dim, features, track_dist_offset);
    else if (solver_type ==  "flat_index")
        labeling = dense_gaec_flat_index(num_nodes, dim, features, track_dist_offset);
    else if (solver_type ==  "inc_nn_flat")
        labeling = dense_gaec_incremental_nn(num_nodes, dim, features, k_inc_nn, "Flat", track_dist_offset);
    else if (solver_type ==  "inc_nn_hnsw")
        labeling = dense_gaec_incremental_nn(num_nodes, dim, features, k_inc_nn, "HNSW", track_dist_offset);
    else
        throw std::runtime_error("Unknown solver type: " + solver_type);
    
    
    if (out_path != "")
    {
        std::ofstream sol_file;
        sol_file.open(out_path);
        std::cout<<"Writing solution to file: "<<out_path<<"\n";
        std::copy(labeling.begin(), labeling.end(), std::ostream_iterator<int>(sol_file, "\n"));
        sol_file.close();
    }
}
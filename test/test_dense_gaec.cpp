#include "test.h"
#include "dense_gaec.h"
#include "dense_gaec_parallel.h"
#include "dense_gaec_adj_matrix.h"
#include "dense_gaec_inc_nn.h"
#include "dense_multicut_utils.h"
#include <random>
#include <iostream>

using namespace DENSE_MULTICUT;

void test_random_problem(const size_t n, const size_t d)
{
    std::cout << "\n[test dense gaec] test random problem with " << n << " features and " << d << " dimensions\n\n";
    std::vector<double> features(n*d);
    std::mt19937 generator(1); // for deterministic behaviour
    std::uniform_real_distribution<float>  distr(-1.0, 1.0);

    for(size_t i=0; i<n*d; ++i)
        features[i] = distr(generator); 

    const auto adj_matrix_cost = labeling_cost<double>(dense_gaec_adj_matrix<double>(n, d, features), n, d, features);
    const auto gaec_bf_cost = labeling_cost<double>(dense_gaec_brute_force<double>(n, d, features), n, d, features);
    const auto gaec_inc_nn_bf_cost = labeling_cost<double>(dense_gaec_inc_nn_brute_force<double>(n, d, features, false, 1, 1), n, d, features);
    test(std::abs(adj_matrix_cost - gaec_bf_cost) < 1e-5);
    test(std::abs(adj_matrix_cost - gaec_inc_nn_bf_cost) < 1e-5);
    // dense_laec_inc_nn(n, d, features, 9);
    // dense_gaec_hnsw(n, d, features);
    // dense_gaec_parallel_flat_index(n, d, features);
    // dense_gaec_parallel_hnsw(n, d, features);
}

int main(int argc, char** argv)
{
    const std::vector<size_t> nr_nodes = {10,20,50,100,1000};
    const std::vector<size_t> nr_dims = {16,32,64,128,256,512,1024};
    for(const size_t n : nr_nodes)
        for(const size_t d : nr_dims)
            test_random_problem(n, d);
}

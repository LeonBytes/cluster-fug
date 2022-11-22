#include "dense_gaec.h"
#include "dense_gaec_parallel.h"
#include "dense_gaec_adj_matrix.h"
#include "dense_gaec_incremental_nn.h"
#include <random>
#include <iostream>

using namespace DENSE_MULTICUT;

void test_random_problem(const size_t n, const size_t d)
{
    std::cout << "\n[test dense gaec] test random problem with " << n << " features and " << d << " dimensions\n\n";
    std::vector<double> features(n*d);
    std::mt19937 generator(0); // for deterministic behaviour
    std::uniform_real_distribution<float>  distr(-1.0, 1.0);

    for(size_t i=0; i<n*d; ++i)
        features[i] = distr(generator); 

    dense_gaec_incremental_nn(n, d, features, 9);
    dense_gaec_adj_matrix(n, d, features);
    dense_gaec_flat_index(n, d, features);
    dense_gaec_hnsw(n, d, features);
    dense_gaec_parallel_flat_index(n, d, features);
    dense_gaec_parallel_hnsw(n, d, features);
}

int main(int argc, char** argv)
{
    const std::vector<size_t> nr_nodes = {10,20,50,100,1000};
    const std::vector<size_t> nr_dims = {16,32,64,128,256,512,1024};
    for(const size_t n : nr_nodes)
        for(const size_t d : nr_dims)
            test_random_problem(n, d);
}

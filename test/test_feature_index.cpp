#include "test.h"
#include "feature_index.h"
#include <random>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace DENSE_MULTICUT;

std::vector<std::tuple<size_t,float>> get_nearest_nodes_brute_force(const size_t n, const size_t d, const std::vector<float>& features)
{
    test(features.size() == n*d);

    std::vector<std::tuple<size_t,float>> nns(n*(n-1));

    for(size_t i=0; i<n; ++i)
        for (size_t j = 0; j < n; ++j)
            if (j < i)
                std::get<0>(nns[i * (n-1) + j]) = j;
            else if (j > i)
                std::get<0>(nns[i * (n-1) + j-1]) = j;

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            double inner_prod = 0.0;
            for (size_t l = 0; l < d; ++l)
                inner_prod += features[i * d + l] * features[j * d + l];
            if (j < i)
                std::get<1>(nns[i * (n - 1) + j]) = inner_prod;
            else if (j > i)
                std::get<1>(nns[i * (n - 1) + j - 1]) = inner_prod;
        }
    }

    for(size_t i=0; i<n; ++i)
    {
        std::sort(nns.begin()+i*(n-1), nns.begin()+(i+1)*(n-1), [] (const auto a, const auto b) {
            return std::get<1>(a) > std::get<1>(b);
        });
    }

    return nns;
}

void test_exact_lookup(const size_t n, const size_t d, const std::string index_str)
{
    std::cout << "test exact lookup for " << n << " elements of dimension " << d << "\n";
    std::vector<float> features(n*d);
    std::mt19937 generator(0); // for deterministic behaviour
    std::uniform_real_distribution<float>  distr(-1.0, 1.0);

    for(size_t i=0; i<n*d; ++i)
        features[i] = distr(generator); 

    feature_index index = feature_index(d, n, features, index_str);
    const std::vector<std::tuple<size_t, float>> nns_brute_force = get_nearest_nodes_brute_force(n, d, features);

    {
        std::vector<faiss::Index::idx_t> all_indices(n); 
        std::iota(all_indices.begin(), all_indices.end(), 0);
        const auto [nns, distances] = index.get_nearest_nodes(all_indices);
        const size_t nr_nns = all_indices.size()/2;
        const auto [nns2, distances2] = index.get_nearest_nodes(all_indices, nr_nns);
        test(nns2.size() == distances2.size() && nns2.size() == nr_nns*all_indices.size());

        for(size_t i=0; i<n; ++i)
        {
            const auto [nn, dist] = index.get_nearest_node(i);
            test(nns[i] == nn);
            test(nns2[i*nr_nns] == nn);
            test(std::abs(distances[i] - dist) < 1e-7*d);
            test(std::abs(dist - index.inner_product(all_indices[i],nn)) < d*1e-7);

            for(size_t l=0; l<nr_nns; ++l)
            {
                if(l+1 < nr_nns && std::abs(distances2[i*nr_nns + l] - distances2[i*nr_nns + l+1]) < 1e-6*d)
                    break;
                test(nns2[i*nr_nns + l] == std::get<0>(nns_brute_force[i*(n-1) + l]));
                test(std::abs(distances2[i*nr_nns + l] - std::get<1>(nns_brute_force[i*(n-1) + l])) < 1e-6*d);
            }
        }
    }

    // remove half the points and test again
    std::vector<faiss::Index::idx_t> remove_indices(n);
    std::iota(remove_indices.begin(), remove_indices.end(), 0);
    std::shuffle(remove_indices.begin(), remove_indices.end(), generator);
    remove_indices.resize(n/2);
    for(size_t i=0; i<remove_indices.size(); ++i)
        index.remove(remove_indices[i]);
    std::sort(remove_indices.begin(), remove_indices.end());

    {
        std::vector<faiss::Index::idx_t> all_indices; 
        for(faiss::Index::idx_t idx=0; idx<n; ++idx)
            if(index.node_active(idx))
                all_indices.push_back(idx);

        const auto [nns, distances] = index.get_nearest_nodes(all_indices);
        const size_t nr_nns = all_indices.size()/2;
        const auto [nns2, distances2] = index.get_nearest_nodes(all_indices, nr_nns);
        test(nns2.size() == distances2.size() && nns2.size() == nr_nns*all_indices.size());

        for(size_t i=0; i<all_indices.size(); ++i)
        {
            const auto [nn, dist] = index.get_nearest_node(all_indices[i]);
            test(nns[i] == nn);
            test(nns2[i*nr_nns] == nn);
            test(std::abs(distances[i] - dist) < 1e-7*d);
            test(std::abs(dist - index.inner_product(all_indices[i],nn)) < d*1e-7);

            size_t l = 0;
            for(size_t k=0; k<n-1; ++k)
            {
                const size_t kth_nn = std::get<0>(nns_brute_force[all_indices[i]*(n-1) + k]);
                if(l < nr_nns && std::count(remove_indices.begin(), remove_indices.end(), kth_nn) == 0)
                {
                    if(l+1 < nr_nns && std::abs(distances2[i*nr_nns+l] - distances2[i*nr_nns+l+1]) < 1e-7*d)
                        break;
                    test(nns2[i*nr_nns + l] == kth_nn);
                    test(std::abs(distances2[i*nr_nns + l] - std::get<1>(nns_brute_force[all_indices[i]*(n-1) + k])) < 1e-7*d);
                    ++l;
                }
            }
        }
    }
}


int main(int argc, char** argv)
{
    const std::vector<size_t> nr_nodes = {10,20,50,100,1000};
    const std::vector<size_t> nr_dims = {16,32,64,128,256,512,1024};
    for(const size_t n : nr_nodes)
        for(const size_t d : nr_dims)
            test_exact_lookup(n, d, "Flat");
}

#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include "dense_features_parser.h"

template<typename REAL>
std::tuple<std::vector<REAL>, size_t, size_t> read_file(const std::string& file_path)
{
    std::ifstream f;
    f.open(file_path);
    if(!f.is_open())
        throw std::runtime_error("Could not open dense multicut input file " + file_path);

    std::string init_line;
    std::getline(f, init_line);
    if(init_line != "FACTORIZED COMPLETE MULTICUT")
        throw std::runtime_error("first line must be 'FACTORIZED COMPLETE MULTICUT'");

    float thresh;
    f >> thresh;
    std::cout<<thresh<<"\n";

    size_t num_nodes, num_dim;
    f >> num_nodes >> num_dim;
    std::cout<<num_nodes<<"\n";
    std::cout<<num_dim<<"\n";

    std::vector<REAL> features(num_nodes * num_dim);
    REAL val;
    size_t index = 0;
    while (f >> val)
    {
        features[index] = val;
        ++index;
    }
    return {features, num_nodes, num_dim};
}

template std::tuple<std::vector<float>, size_t, size_t> read_file(const std::string& file_path);
template std::tuple<std::vector<double>, size_t, size_t> read_file(const std::string& file_path);
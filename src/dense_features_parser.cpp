#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include "dense_features_parser.h"

template<typename REAL>
std::tuple<std::vector<REAL>, size_t, size_t> read_file(const std::string& file_path)
{
    std::ifstream f;
    f.open(file_path);
    if(!f.is_open())
        throw std::runtime_error("Could not open dense multicut input file " + file_path);

    size_t num_nodes, num_dim;
    f >> num_nodes >> num_dim;

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
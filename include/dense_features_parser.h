#include <string>
#include <vector>
#include <tuple>

template<typename REAL>
std::tuple<std::vector<REAL>, size_t, size_t, REAL> read_file(const std::string& filename);

#include "feature_index.h"
#include "feature_index_hnswlib.h"
#include "time_measure_util.h"
#include <thread>
#include <atomic>
#include <cassert>
#include <mutex>

namespace DENSE_MULTICUT {

    /*
    * replacement for the openmp '#pragma omp parallel for' directive
    * only handles a subset of functionality (no reductions etc)
    * Process ids from start (inclusive) to end (EXCLUSIVE)
    *
    * The method is borrowed from nmslib
    */
    template<class Function>
    inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
        if (numThreads <= 0) {
            numThreads = std::thread::hardware_concurrency();
        }

        if (numThreads == 1) {
            for (size_t id = start; id < end; id++) {
                fn(id, 0);
            }
        } else {
            std::vector<std::thread> threads;
            std::atomic<size_t> current(start);

            // keep track of exceptions in threads
            // https://stackoverflow.com/a/32428427/1713196
            std::exception_ptr lastException = nullptr;
            std::mutex lastExceptMutex;

            for (size_t threadId = 0; threadId < numThreads; ++threadId) {
                threads.push_back(std::thread([&, threadId] {
                    while (true) {
                        size_t id = current.fetch_add(1);

                        if ((id >= end)) {
                            break;
                        }

                        try {
                            fn(id, threadId);
                        } catch (...) {
                            std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                            lastException = std::current_exception();
                            /*
                            * This will work even when current is the largest value that
                            * size_t can fit, because fetch_add returns the previous value
                            * before the increment (what will result in overflow
                            * and produce 0 instead of current + 1).
                            */
                            current = end;
                            break;
                        }
                    }
                }));
            }
            for (auto &thread : threads) {
                thread.join();
            }
            if (lastException) {
                std::rethrow_exception(lastException);
            }
        }
    }

    feature_index_hnswlib::feature_index_hnswlib(const size_t _d, const size_t n, const std::vector<float>& _features, const std::string& _index_str, const bool track_dist_offset)
    :feature_index<float>(_d, n, _features, track_dist_offset), index_str(_index_str)
    {
        hnswlib::InnerProductSpace* space = new hnswlib::InnerProductSpace(this->d);
        if (index_str == "hnswlib_brute_force")
            index  = new hnswlib::BruteforceSearch<float>(space, (size_t) 2 * n);
        else
            index = new hnswlib::HierarchicalNSW<float>(space, (size_t) 2 * n);

        for (size_t i = 0; i < n; ++i)
            index->addPoint(this->features.data() + this->d * i, i);
    }

    std::tuple<idx_t, float> feature_index_hnswlib::get_nearest_node(const idx_t node) const
    {
        const idx_t node_internal = this->get_orig_to_internal_node_mapping(node);

        const void* p = this->features.data() + node_internal * this->d;
        std::vector<std::pair<float, size_t>> res = index->searchKnnCloserFirst(p, 2);
        return {res[1].second, 1.0 - res[1].first};
    }

    std::tuple<std::vector<idx_t>, std::vector<float>> feature_index_hnswlib::get_nearest_nodes(const std::vector<idx_t>& nodes, const size_t k) const
    {
        const auto num_query_nodes = nodes.size();
        int num_threads = std::thread::hardware_concurrency();
        if(num_query_nodes <= num_threads * 4)
            num_threads = 1;

        std::vector<float> final_distances(num_query_nodes * k, -1.0);
        std::vector<idx_t> final_ids(num_query_nodes * k);

        ParallelFor(0, num_query_nodes, num_threads, [&](size_t q, size_t threadId) {
            const size_t node_internal = this->get_orig_to_internal_node_mapping(nodes[q]);
            const void* p = this->features.data() + node_internal * this->d;
            std::priority_queue<std::pair<float, size_t>> result = index->searchKnn(p, k + 1);
            assert(result.size() == 2);
            int out_index = 0;
            for (int i = k; i >= 0; i--) {
                auto &result_tuple = result.top();
                if (result_tuple.second != node_internal && out_index < k)
                {
                    final_distances[q * k + out_index] = 1.0 - result_tuple.first;
                    final_ids[q * k + out_index] = this->get_internal_to_orig_node_mapping(result_tuple.second);
                    ++out_index;
                }
                result.pop();
            }
        });
        return {final_ids, final_distances};
    }

    std::tuple<std::vector<idx_t>, std::vector<float>> feature_index_hnswlib::get_nearest_nodes(const std::vector<idx_t>& nodes) const
    {
        return this->get_nearest_nodes(nodes, 1);
    }

    feature_index_hnswlib::~feature_index_hnswlib()
    {
        delete index;
    }
}
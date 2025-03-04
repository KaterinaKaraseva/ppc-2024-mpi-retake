#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/chernova_n_word_count/include/ops_mpi.hpp"

std::vector<char> chernova_n_word_count_mpi::GenerateWordsPerf(int k) {
  const std::string words[] = {"one", "two", "three"};
  const int word_array_size = sizeof(words) / sizeof(words[0]);

  std::string result;

  for (int i = 0; i < k; ++i) {
    result += words[i % word_array_size];
    if (i < k - 1) {
      result += ' ';
    }
  }

  return {result.begin(), result.end()};
}

const int k_ = 100000;
namespace {
std::vector<char> test_data_parallel = chernova_n_word_count_mpi::GenerateWordsPerf(k_);
}  // namespace

TEST(chernova_n_word_count_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<char> in = test_data_parallel;
  std::vector<int> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  auto test_task_mpi = std::make_shared<chernova_n_word_count_mpi::TestMPITaskParallel>(task_data_mpi);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 1000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(out[0], k_);
  }
}

TEST(chernova_n_word_count_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<char> in = test_data_parallel;
  std::vector<int> out(1, 0);

  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    task_data_mpi->inputs_count.emplace_back(in.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  auto test_task_mpi = std::make_shared<chernova_n_word_count_mpi::TestMPITaskParallel>(task_data_mpi);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 1000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_EQ(out[0], k_);
  }
}
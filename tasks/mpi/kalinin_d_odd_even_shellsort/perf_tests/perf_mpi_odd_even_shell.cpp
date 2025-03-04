#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/kalinin_d_odd_even_shellsort/include/header_mpi_odd_even_shell.hpp"

TEST(kalinin_d_odd_even_shellsort_mpi, test_pipeline_run) {
  const int n = 3000000;

  boost::mpi::communicator world;

  // Create data
  std::vector<int> arr;
  std::vector<int> out;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr.resize(n);
    out.resize(n);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data_mpi->inputs_count.emplace_back(arr.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  auto test_mpi_task_parallel = std::make_shared<kalinin_d_odd_even_shell_mpi::OddEvenShellMpi>(task_data_mpi);
  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  // Create Perf analyzer
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    std::ranges::sort(arr.begin(), arr.end());
    ASSERT_EQ(arr, out);
  }
}

TEST(kalinin_d_odd_even_shellsort_mpi, test_task_run) {
  const int n = 3000000;

  boost::mpi::communicator world;

  // Create data
  std::vector<int> arr;
  std::vector<int> out;
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    arr.resize(n);
    out.resize(n);
    task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(arr.data()));
    task_data_mpi->inputs_count.emplace_back(arr.size());
    task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_mpi->outputs_count.emplace_back(out.size());
  }

  auto test_mpi_task_parallel = std::make_shared<kalinin_d_odd_even_shell_mpi::OddEvenShellMpi>(task_data_mpi);
  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_mpi_task_parallel);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  // Create Perf analyzer
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    std::ranges::sort(arr.begin(), arr.end());
    ASSERT_EQ(arr, out);
  }
}

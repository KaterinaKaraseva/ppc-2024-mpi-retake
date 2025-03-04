#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/karaseva_e_reduce/include/ops_mpi.hpp"

TEST(karaseva_e_reduce_mpi, test_pipeline_run) {
  constexpr int kCount = 10000;

  // Create data
  std::vector<int> in(kCount * kCount, 1);
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<unsigned char*>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<unsigned char*>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_mpi = std::make_shared<karaseva_e_reduce_mpi::TestTaskMPI<int>>(task_data_mpi);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  const int expected_sum = kCount * kCount;
  EXPECT_EQ(out[0], expected_sum);
}

TEST(karaseva_e_reduce_mpi, test_task_run) {
  constexpr int kCount = 10000;

  // Create data
  std::vector<int> in(kCount * kCount, 1);
  std::vector<int> out(1, 0);

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<unsigned char*>(in.data()));
  task_data_mpi->inputs_count.emplace_back(in.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<unsigned char*>(out.data()));
  task_data_mpi->outputs_count.emplace_back(out.size());

  // Create Task
  auto test_task_mpi = std::make_shared<karaseva_e_reduce_mpi::TestTaskMPI<int>>(task_data_mpi);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  // Checking that the output result is equal to the sum of all the elements
  const int expected_sum = kCount * kCount;
  EXPECT_EQ(out[0], expected_sum);
}

#define OMPI_SKIP_MPICXX